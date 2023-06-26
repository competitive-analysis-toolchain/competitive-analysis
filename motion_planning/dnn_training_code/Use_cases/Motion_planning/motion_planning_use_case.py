from pyclbr import Function
import sys
import os
import csv
import tensorflow as tf
import pandas as pd
from tensorflow import keras
#from keras import layers
from tensorflow.keras import layers
from Use_Case_class import Use_Case
from rockit import casadi_helpers
import keras_tuner as kt
from Use_Case_class import Data_set
from rockit import *
from rockit import casadi_helpers
import matplotlib.pyplot as plt
from pylab import *
from casadi import vertcat, horzcat, sumsqr
from random import choices
from typing import List, Dict

class Motion_planning_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['X','Y','THETA']
        self.NN_labels_input = ['X','Y','THETA']
        self.labels_output = ['DELTA','V', 'Timestep']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [30, 29, 28]
        self.timesteps = [1.5]*len(self.perfect_paths)
        self.custom_objects = {}

        [ocp, delta, V] = initialize_ocp(0, 0, 0)
        self.simulation_function = ocp._method.discrete_system(ocp)

        pathpoints = 30
        ref_path = {}
        ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
        ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
        wp = horzcat(ref_path['x'], ref_path['y']).T

        self.ref_path = ref_path
        
    def give_hypermodel(self, normalizer, train_features):
        """This function gives the hyper model"""
        hyper_model = Motion_planning_HyperModel(normalizer, train_features)
        return hyper_model

    def give_NN_model(self, normalizer, hyper_parameters, train_features):
        """This function gives the NN model"""
        model = build_and_compile_model(normalizer, hyper_parameters, train_features)
        return model

    def give_expert_actions(self, input):
        """This function gives expert actions for a list of states"""
        output_delta = []
        output_V = []
        output_timestep = []

        for idx in range(len(input[0])):
            [delta, V, timestep] = self.give_expert_action([input[0][idx], input[1][idx], input[2][idx]])

            output_delta.append(delta)
            output_V.append(V)
            output_timestep.append(timestep)

        return [output_delta, output_V, output_timestep]

    def give_expert_action(self, input, iterations=0):
        """This function gives an expert action for a state"""
        try:
            [ocp, delta, V] = initialize_ocp(input[0], input[1], input[2])
            sol = ocp.solve()
            tsol, deltasol = sol.sample(delta, grid='control')
            tsol, Vsol = sol.sample(V, grid='control')
            return [deltasol[0], Vsol[0], tsol[1]]
        except:
            return [False, False, False]

    def give_next_state(self, input, control_variables, time_step, idx = 0):
        """This function gives the next state for a state and a control variables"""
        current_X = vertcat(input[0], input[1], input[2])
        current_U = vertcat(control_variables[0], control_variables[1])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=control_variables[2])["xf"]
        system_result = casadi_helpers.DM2numpy(system_result, [2,1]) 

        return [system_result[0], system_result[1], system_result[2]]
    
    def give_NN_data(self, data_set):
        """This function transform the data to the NN data"""
        old_data_set_input_rows = data_set.give_rows(give_input = True, give_output = False)
        old_data_set_output_cols = data_set.give_columns(give_input = False, give_output = True)

        NN_data_set_input_rows = []
        for old_data_set_input_row in old_data_set_input_rows:
            NN_data_set_input_rows.append(self.give_NN_data_row(old_data_set_input_row))

        NN_Data_set = Data_set(len(self.NN_labels_input), len(self.labels_output))
        NN_Data_set.load_data_from_row_list(NN_data_set_input_rows, load_only_output = False)
        NN_Data_set.load_data_from_col_list(old_data_set_output_cols, load_only_output = True)

        return NN_Data_set
    
    def give_NN_data_row(self, data_row):
        """This function gives a NN data row"""
        amount_of_ref_path_points = 0

        new_data_row = data_row.copy()

        ref_point_array = get_ref_points(data_row[0], data_row[1], amount_of_ref_path_points)

        for ref_point in ref_point_array:
            new_data_row.append(ref_point)

        return new_data_row

    def sample_k_start_points(self, k):
        """This function gives k start points"""
        start_points = []
        end_iterations = []

        start_index_list = choices([0, 1, 2], k=k)

        for start_index in start_index_list:
            start_point = [self.ref_path['x'][start_index], self.ref_path['y'][start_index], 0]
            start_points.append(start_point)
            
            end_iterations.append(30-start_index)
        
        return start_points, end_iterations

class Motion_planning_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization, train_features: pd.DataFrame):
        self.normalizer = norm
        self.train_features = train_features

    def build(self, hp: kt.engine.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        # input_layer = keras.Input(shape=(len(self.train_features.columns),))
        # #x = self.normalizer(input_layer)
        # x=input_layer
        # for _ in range(int(hidden_layers)):
        #     x = layers.Dense(hp_units, activation=hp_activation_function)(x)

        # delta_output = layers.Dense(1, name = 'delta')(x)
        # v_output = layers.Dense(1, name = 'v')(x)
        # time_step_output = layers.Dense(1, name = 'time_step')(x)

        # model = tf.keras.Model(inputs=input_layer, outputs=[delta_output, v_output, time_step_output])

        # hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        # model.compile(optimizer=tf.keras.optimizers.Adam(hp_learning_rate), loss={'delta': 'mse', 'v': 'mse', 'time_step': 'mse'})

        model = keras.Sequential()
        inputs = tf.keras.Input(shape=[len(self.train_features.columns),])
        model.add(inputs)
        #model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(3))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model

def build_and_compile_model(norm: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
    """This function gives the NN model"""
    # input_layer = keras.Input(shape=(len(train_features.columns),))
    # #x = norm(input_layer)
    # x=input_layer
   
    # for _ in range(int(hyper_parameters[3])):
    #     x = layers.Dense(hyper_parameters[0], activation=hyper_parameters[1])(x)

    # delta_output = layers.Dense(1, name = 'delta')(x)
    # v_output = layers.Dense(1, name = 'v')(x)
    # time_step_output = layers.Dense(1, name = 'time_step')(x)

    # model = tf.keras.Model(inputs=input_layer, outputs=[delta_output, v_output, time_step_output])

    # model.compile(optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]), loss={'delta': 'mse', 'v': 'mse', 'time_step': 'mse'})
    model = keras.Sequential()
    inputs = tf.keras.Input(shape=[len(train_features.columns),])
    model.add(inputs)
    #model.add(norm)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(3))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]))
    return model


def initialize_ocp(x_init: float, y_init: float, theta_init: float) -> List[Function]:
    """Function that initialize the ocp for the motion plannning"""
   # -------------------------------
    # Problem parameters
    # -------------------------------

    L       = 1             # bicycle model length
    nx      = 3             # the system is composed of 3 states
    nu      = 2             # the system has 2 control inputs
    N       = 10            # number of control intervals


    # -------------------------------
    # Set OCP
    # -------------------------------

    ocp = Ocp(T=FreeTime(10.0))

    # Bicycle model

    # Define states
    x     = ocp.state()
    y     = ocp.state()
    theta = ocp.state()

    # Defince controls
    delta = ocp.control()
    V     = ocp.control(order=0)

    # Specify ODE
    ocp.set_der(x,      V*cos(theta))
    ocp.set_der(y,      V*sin(theta))
    ocp.set_der(theta,  V/L*tan(delta))

    # Define parameter
    X_0 = ocp.parameter(nx)

    # Initial constraints
    X = vertcat(x, y, theta)
    ocp.subject_to(ocp.at_t0(X) == X_0)

    # Initial guess
    ocp.set_initial(x,      0)
    ocp.set_initial(y,      0)
    ocp.set_initial(theta,  0)

    ocp.set_initial(V,    0.5)
    #ocp.set_initial(delta,  -pi/6)

    # Path constraints
    ocp.subject_to( 0 <= (V <= 1) )
    #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
    ocp.subject_to( -pi/6 <= (delta <= pi/6) )

    # Minimal time
    # ocp.add_objective(0.50*ocp.T)

    # Define physical path parameter
    waypoints = ocp.parameter(2, grid='control')
    waypoint_last = ocp.parameter(2)
    p = vertcat(x,y)

    # waypoints = ocp.parameter(3, grid='control')
    # waypoint_last = ocp.parameter(3)
    # p = vertcat(x,y,theta)

    ocp.add_objective(ocp.sum(sumsqr(p-waypoints), grid='control'))
    ocp.add_objective(sumsqr(ocp.at_tf(p)-waypoint_last))

    # Pick a solution method
    options = {"ipopt": {"print_level": 0}}
    options["expand"] = True
    options["print_time"] = False
    ocp.solver('ipopt', options)

    # Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

    # Define reference path
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = horzcat(ref_path['x'], ref_path['y']).T

    # -------------------------------
    # Solve the OCP wrt a parameter value (for the first time)
    # -------------------------------

    # Set initial value for states
    current_X = vertcat(x_init, y_init, theta_init)
    ocp.set_value(X_0, current_X)

    # First waypoint is current position
    index_closest_point = find_closest_point(current_X[:2], ref_path, 0)

    # Create a list of N waypoints
    current_waypoints = get_current_waypoints(index_closest_point, wp, N, dist=6)

    # Set initial value for waypoint parameters
    ocp.set_value(waypoints,current_waypoints[:,:-1])
    ocp.set_value(waypoint_last,current_waypoints[:,-1])

    ocp.set_value(X_0, current_X)

    return [ocp, delta, V]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""
    with open("Use_cases/Motion_planning/MPC_data/MPC_x_motion_planning.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_x = []
        for row in csvreader:
            MPC_x.append([float(i) for i in row])
        
    with open("Use_cases/Motion_planning/MPC_data/MPC_y_motion_planning.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_y = []
        for row in csvreader:
            MPC_y.append([float(i) for i in row])

    with open("Use_cases/Motion_planning/MPC_data/MPC_theta_motion_planning.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_theta = []
        for row in csvreader:
            MPC_theta.append([float(i) for i in row])
  
    perfect_paths = []
    
    for idx in range(len(MPC_x)):
        perfect_path = Data_set(3, 0)
        perfect_path.load_data_from_col_list([MPC_x[idx], MPC_y[idx], MPC_theta[idx]])
        perfect_paths.append(perfect_path)

    return perfect_paths

#---------------------------------------------------#
# Helper functions for the motion planning use case #
#---------------------------------------------------#

def get_ref_points(x: float, y: float, amount_of_ref_path_points: int) -> List[float]:
    """"this function gives an array with amount_of_ref_path_points points forward of the current position (x,y)"""
    refpoint_array=[]

    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10

    close_index = find_closest_point([x, y], ref_path, 0)
    for steps_foward in range(amount_of_ref_path_points):
        if close_index+steps_foward<31:
            x_dis = ref_path['x'][close_index+steps_foward]-x
            y_dis = ref_path['y'][close_index+steps_foward]-y
        else:
            x_dis = ref_path['x'][30]-x
            y_dis = ref_path['y'][30]-y
        refpoint_array.append(x_dis)
        refpoint_array.append(y_dis)
    
    return refpoint_array

# Find closest point on the reference path compared witch current position
def find_closest_point(pose, reference_path: Dict, start_index: int) -> int:
    # x and y distance from current position (pose) to every point in 
    # the reference path starting at a certain starting index
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]
    # Index of closest point by Pythagoras theorem
    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    #print('find_closest_point results in', index_closest)
    return index_closest

# Return the point on the reference path that is located at a certain distance 
# from the current position
def index_last_point_fun(start_index: int, wp, dist: int) -> int:
    pathpoints = wp.shape[1]
    # Cumulative distance covered
    cum_dist = 0
    # Start looping the index from start_index to end
    for i in range(start_index, pathpoints-1):
        # Update comulative distance covered
        cum_dist += np.linalg.norm(wp[:,i] - wp[:,i+1])
        # Are we there yet?
        if cum_dist >= dist:
            #print('cumdist >= dist:', cum_dist)
            return i + 1
    # Desired distance was never covered, -1 for zero-based index
    #print('cum_dist >= dist:', cum_dist)
    return pathpoints - 1

# Create a list of N waypoints
def get_current_waypoints(start_index: int, wp, N: int, dist: int):
    # Determine index at reference path that is dist away from starting point
    last_index = index_last_point_fun(start_index, wp, dist)
    # Calculate amount of indices between last and start point
    delta_index = last_index - start_index
    # Dependent on the amount of indices, do
    if delta_index >= N: 
        # There are more than N path points available, so take the first N ones
        index_list = list(range(start_index, start_index+N+1))
        #print('index list with >= N points:', index_list)
    else:
        # There are less than N path points available, so add the final one multiple times
        index_list = list(range(start_index, last_index)) + [last_index]*(N-delta_index+1)
        #print('index list with < N points:', index_list)
    return wp[:,index_list]

#-------------------------------------------#
# Functions to disable and restore printing #
#-------------------------------------------#

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    print('h')
