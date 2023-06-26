import sys
import os
import csv
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from casadi import vertcat
from keras import layers
from Use_Case_class import Use_Case
from casadi import Function
from rockit import casadi_helpers
import keras_tuner as kt
from Use_Case_class import Data_set
from random import uniform
from typing import List

#sys.path.insert(0, 'C:\\Users\\Stijn\\Documents\\Dirac\\dirac2\\SCARA_lib\\wp5\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir')
sys.path.insert(0, 'C:\\Users\\Werk\\dirac\\dirac\\SCARA_lib\\wp5_scara-ugent_usecase\\wp5_scara-ugent_usecase\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir')
from impact import Impact
#sys.path.insert(0, 'C:\\Users\\Stijn\\Documents\\Dirac\\dirac2')
sys.path.insert(0, 'C:\\Users\\Werk\\dirac\\dirac')

class SCARA_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['X','Y','VX','VY']
        self.NN_labels_input = ['X','Y','VX','VY']
        self.labels_output = ['U1','U2']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [99]*len(self.perfect_paths)
        self.timesteps = [0.008]*len(self.perfect_paths)
        self.custom_objects = {}

        [impact, func] = initialize_ocp()
        self.MPC_function = impact
        self.simulation_function = func
        
    def give_hypermodel(self, normalizer: tf.keras.layers.experimental.preprocessing.Normalization, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        hyper_model = SCARA_HyperModel(normalizer, train_features)
        return hyper_model

    def give_NN_model(self, normalizer: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        model = build_and_compile_model(normalizer, hyper_parameters, train_features)
        return model

    def give_expert_actions(self, input: List[List[float]]) -> List[List[float]]:
        """This function gives expert actions for a list of states"""
        output_U1 = []
        output_U2 = []

        for idx in range(len(input[0])):
            [U1, U2] = self.give_expert_action([input[0][idx], input[1][idx], input[2][idx], input[3][idx]]) 

            output_U1.append(U1)
            output_U2.append(U2)

        return [output_U1, output_U2]

    def give_expert_action(self, input: List[float], iterations=0) -> List[float]:
        """This function gives an expert action for a state"""
        
        try:
            blockPrint()
            self.MPC_function.set("x_current", self.MPC_function.ALL, 0, self.MPC_function.FULL, [[input[0]], [input[1]], [input[2]], [input[3]]])
            self.MPC_function.solve()
            u = self.MPC_function.get("u_opt", self.MPC_function.ALL, 0, self.MPC_function.FULL)
            enablePrint()
            return [u[0][0], u[1][0]]
        except:
            return [False, False]

    def give_next_state(self, input: List[float], control_variables: List[float], time_step: float, idx: int = 0) -> List[float]:
        """This function gives the next state for a state and control variables"""
        system_result = self.simulation_function(input, control_variables, time_step)
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        return [system_result[0], system_result[1], system_result[2], system_result[3]]

    def sample_k_start_points(self, k: int):
        """This function gives k start points"""
        start_points = []

        for _ in range(k):
            start_x = uniform(0.09,0.11)
            start_y = uniform(0.39,0.41)

            start_point = [start_x, start_y, 0, 0]
            start_points.append(start_point)
        
        end_iterations = [99]*k
        
        return start_points, end_iterations


class SCARA_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization, train_features: pd.DataFrame):
        self.normalizer = norm
        self.train_features = train_features

    def build(self, hp: kt.engine.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh','swish'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        input_layer = keras.Input(shape=(len(self.train_features.columns),))
        x = self.normalizer(input_layer)
        for _ in range(int(hidden_layers)):
            x = layers.Dense(hp_units, activation=hp_activation_function)(x)

        u1_output = layers.Dense(1, name = 'u1')(x)
        u2_output = layers.Dense(1, name = 'u2')(x)
        
        model = tf.keras.Model(inputs=input_layer, outputs=[u1_output, u2_output])

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(hp_learning_rate), loss={'u1': 'mse', 'u2': 'mse'})
        return model

def build_and_compile_model(norm: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
    """This function gives the NN model"""
    input_layer = keras.Input(shape=(len(train_features.columns),))
    x = norm(input_layer)
    for _ in range(int(hyper_parameters[3])):
        x = layers.Dense(hyper_parameters[0], activation=hyper_parameters[1])(x)

    u1_output = layers.Dense(1, name = 'u1')(x)
    u2_output = layers.Dense(1, name = 'u2')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[u1_output, u2_output])

    model.compile(optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]), loss={'u1': 'mse', 'u2': 'mse'})
    return model


def initialize_mcp():
    #impact = Impact("scara_ipopt_cartesian_doubleinteg",src_dir="C:\\Users\\Stijn\\Documents\\Dirac\\dirac2\\SCARA_lib\\wp5\\impact_workflow\\libraries\\")
    #func = Function.load("C:\\Users\\Stijn\\Documents\\Dirac\\dirac2\\SCARA_lib\\wp5\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir\\integrate_scara_ipopt_cartesian_doubleinteg.casadi")

    impact = Impact("scara_ipopt_cartesian_doubleinteg",src_dir="C:\\Users\\Werk\\dirac\\dirac\\SCARA_lib\\wp5_scara-ugent_usecase\\wp5_scara-ugent_usecase\\impact_workflow\\libraries\\")
    func = Function.load("C:\\Users\\Werk\\dirac\\dirac\\SCARA_lib\\wp5_scara-ugent_usecase\\wp5_scara-ugent_usecase\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir\\integrate_scara_ipopt_cartesian_doubleinteg.casadi")

    # Example: how to set a parameter
    val_p_end = [0.299999999999999989,0.225000000000000006]
    impact.set("p", "p_end", impact.EVERYWHERE, impact.FULL, val_p_end)
    return [impact, func]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""
    with open("Use_cases\Scara\MPC_data\MPC_x_scara.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_x = []
        for row in csvreader:
            MPC_x.append([float(i) for i in row])
        
    with open("Use_cases\Scara\MPC_data\MPC_y_scara.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_y = []
        for row in csvreader:
            MPC_y.append([float(i) for i in row])

    with open("Use_cases\Scara\MPC_data\MPC_vx_scara.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_vx = []
        for row in csvreader:
            MPC_vx.append([float(i) for i in row])
    
    with open("Use_cases\Scara\MPC_data\MPC_vy_scara.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_vy = []
        for row in csvreader:
            MPC_vy.append([float(i) for i in row])
    
    perfect_paths = []
    
    for idx in range(len(MPC_x)):
        perfect_path = Data_set(4, 0)
        perfect_path.load_data_from_col_list([MPC_x[idx], MPC_y[idx], MPC_vx[idx], MPC_vy[idx]])
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
    use_case = SCARA_Use_case()
    use_case.set_self_parameters()
    print('h')