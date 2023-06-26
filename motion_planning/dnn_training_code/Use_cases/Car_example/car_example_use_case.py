import sys
import os
import csv
import pandas as pd
from rockit import *
from pylab import *
from casadi import vertcat
import tensorflow as tf
#import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow import keras
#from keras import layers
from tensorflow.keras import layers
from Use_Case_class import Use_Case
from casadi import Function
from rockit import casadi_helpers
import keras_tuner as kt
from Use_Case_class import Data_set
from typing import List

class Car_example_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['POS','V']
        self.NN_labels_input = ['POS','V']
        self.labels_output = ['F']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [20]*len(self.perfect_paths)
        self.timesteps = [2.0]*len(self.perfect_paths)
        self.custom_objects = {}
        self.MPC_function = False

        [ocp, _, _, _] = initialize_ocp(0, 0, 20)
        Sim_cart_pole_dyn = ocp._method.discrete_system(ocp)
        self.simulation_function = Sim_cart_pole_dyn
        
    def give_hypermodel(self, normalizer: tf.keras.layers.experimental.preprocessing.Normalization, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        hyper_model = Car_example_HyperModel(normalizer)
        return hyper_model

    def give_NN_model(self, normalizer: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        model = build_and_compile_model(normalizer, hyper_parameters)
        return model

    def give_expert_actions(self, input: List[List[float]]) -> List[List[float]]:
        """This function gives expert actions for a list of states"""
        output_F = []

        for idx in range(len(input[0])):
            blockPrint()
            [F] = self.give_expert_action([input[0][idx], input[1][idx]], 20-idx)
            enablePrint() 
            output_F.append(F)

        return [output_F]

    def give_expert_action(self, input: List[float], iterations) -> List[float]:
        """This function gives an expert action for a state"""
        [ocp, F,_,_] = initialize_ocp(input[0], input[1], iterations)
        blockPrint()
        try:
            sol = ocp.solve()
            tsol, usol = sol.sample(F, grid='control')
            enablePrint() 
            return [usol[0]]
        except:
            enablePrint() 
            return [False]

    def give_next_state(self, input: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(input[0], input[1])
        current_U = vertcat(control_variables[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])
            
        return [system_result[0], system_result[1]]

    def sample_k_start_points(self, k: int):
        """This function gives k start points"""
        start_points = []

        for _ in range(k):
            start_pos = uniform(0, 200)

            start_point = [start_pos, 0]
            start_points.append(start_point)
        
        end_iterations = [20]*k
        
        return start_points, end_iterations

class Car_example_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization):
        self.normalizer = norm

    def build(self, hp: kt.engine.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh', 'swish'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        model = keras.Sequential()
        model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(1))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model

def build_and_compile_model(norm: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List) -> tf.keras.Sequential:
    """This function gives the NN model"""

    model = keras.Sequential()
    model.add(norm)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(1))
 
    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]))
    return model

def initialize_ocp(position: float, speed: float, amount_of_control_steps: int):
    """Function that initialize the ocp for the car example"""
    ocp = Ocp(T=amount_of_control_steps*2.0)
  
    # Define constants
    m = 500.0
    c = 2
    d = 1000
    F_max = 2500

    # Define states
    p = ocp.state()
    v = ocp.state()

    # Defince controls
    F = ocp.control()

    # Specify ODE
    ocp.set_der(p, v)
    ocp.set_der(v, 1/m * (F - c * v**2))

    # Lagrange objective
    ocp.add_objective(ocp.T)

    # Path constraints
    ocp.subject_to(-F_max <= (F<= F_max))
    ocp.subject_to(v >= 0)

    # Initial constraints
    ocp.subject_to(ocp.at_t0(p)==position)
    ocp.subject_to(ocp.at_t0(v)==speed)

    # End constraints
    ocp.subject_to(ocp.at_tf(p)==d)
    ocp.subject_to(ocp.at_tf(v)==0)

    # Pick a solver
    ocp.solver('ipopt')
    # Choose a solution method
    ocp.method(MultipleShooting(N=amount_of_control_steps,M=1,intg='rk'))

    return [ocp, F, v, p]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""

    with open("Use_cases/Car_example/MPC_data/MPC_position_car_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_pos = []
        for row in csvreader:
            MPC_pos.append([float(i) for i in row])
        
    with open("Use_cases/Car_example/MPC_data/MPC_speed_car_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_speed = []
        for row in csvreader:
            MPC_speed.append([float(i) for i in row])

    perfect_paths = []
    
    for idx in range(len(MPC_pos)):
        perfect_path = Data_set(2, 0)
        perfect_path.load_data_from_col_list([MPC_pos[idx], MPC_speed[idx]])
        perfect_paths.append(perfect_path)

    return perfect_paths

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
    use_case = Car_example_Use_case()
    use_case.set_self_parameters()
    print('h')
