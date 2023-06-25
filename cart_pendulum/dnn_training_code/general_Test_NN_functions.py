from rockit import *
import matplotlib.pyplot as plt
from pylab import *
import csv
import sys, os
import tensorflow as tf
import pandas as pd
import shutil
import numpy as np
from Use_Case_class import Use_Case, Data_set, give_list_list
from typing import List
from general_NN_training_functions import load_NN_from_weights, get_and_process_data
from general_hyper_functions import get_hyperparameters_from_file
from Use_cases.Cart_pole.cart_pole_use_case import Cart_pole_Use_case

def Evaluate_NN_with_MPC(use_case: Use_Case, saved_NN_weights: str, hyper_parameters_file: str, saved_Dataset: str, save_place_new_dataset: str, save_place_MSE: str, step_size_polish: int, clone_existing_data: bool =True, generate_extra_data: bool =True):
    
    if clone_existing_data:
        shutil.copyfile(saved_Dataset, save_place_new_dataset)

    MSE_result_lists=give_list_list(len(use_case.labels_input))
    perfect_paths = use_case.perfect_paths

    [train_features, _, _, _] = get_and_process_data(saved_Dataset, use_case.NN_labels_input, use_case.labels_output)
    hyper_parameters = get_hyperparameters_from_file(hyper_parameters_file)
    NN = load_NN_from_weights(use_case, hyper_parameters, train_features, saved_NN_weights)

    for idx, perfect_path in enumerate(perfect_paths):
        print(str(idx+1)+'/'+str(len(perfect_paths)))

        Test_data_set = generate_data_for_start_position(use_case, NN, perfect_path.give_input_row(0), use_case.timesteps[idx], save_place_new_dataset, use_case.end_iterations[idx], generate_extra_data, step_size_polish)

        get_MSE(Test_data_set, perfect_path, MSE_result_lists)

    save_MSE(MSE_result_lists, use_case.labels_input, save_place_MSE)


def generate_data_for_start_position(use_case: Use_Case, NN: tf.keras.Sequential, start_position: List[float], time_step: float, save_place_new_dataset: str, end_iterations: List[int], generate_extra_data: bool, step_size_polish: int) -> Data_set:
    """For a start position generates new data by testing the NN and the asking the MPC what it would do in the positions and speeds that
    the NN reaches then it saves this data"""
   
    Test_data_set = test_NN(use_case, NN, start_position, time_step, end_iterations)
    
    if generate_extra_data:
        blockPrint()
        extra_data_set= get_extra_data_polish(use_case, Test_data_set, time_step, end_iterations, step_size_polish)
        enablePrint()
    
        NN_Data_set = use_case.give_NN_data(extra_data_set)
        NN_Data_set.save_data(save_place_new_dataset)

    return Test_data_set

def test_NN(use_case: Use_Case, NN: tf.keras.Sequential, input: List[float], time_step: float, end_iteration: int) -> Data_set:
    """given a start position calculates in what positions and speeds the car gets while being controlled by the NN"""

    Test_data_set = Data_set(len(use_case.labels_input), len(use_case.labels_output))

    for idx in range(end_iteration):
        control_action = get_control_action_from_NN(use_case, NN, input)
        input = use_case.give_next_state(input, control_action, time_step, idx)

        Test_data_set.add_rows([input])
        
    return Test_data_set

def get_control_action_from_NN(use_case: Use_Case, NN: tf.keras.Sequential, input: List[float]) -> List[float]:
    """This function gives a control action from the NN"""
    input_row = use_case.give_NN_data_row(input)

    NN_input = pd.DataFrame([input_row], columns = use_case.NN_labels_input)
    NN_output = NN(NN_input)

    NN_output_list = []

    if len(use_case.labels_output) == 1:
        NN_output_list.append(NN_output.numpy()[0][0])
    else:
        for NN_output_parameter in NN_output:
            NN_output_list.append(NN_output_parameter.numpy()[0][0])

    return NN_output_list

# def get_extra_data(use_case: Use_Case, Test_data_set: Data_set):
#     """Function that makes extra training by seeing what the MPC would do in the positions and speeds that the NN gets to"""
#     columns = Test_data_set.give_columns(give_input=True, give_output=False)
#     control_variables = use_case.give_expert_actions(columns)
    
#     extra_data_set = Data_set(len(use_case.labels_input), len(use_case.labels_output))
#     extra_data_set.load_data_from_col_list(columns, load_only_output = False)
#     extra_data_set.load_data_from_col_list(control_variables, load_only_output = True)
#     extra_data_set.delete_False_outputs()

#     return extra_data_set

def get_extra_data_polish(use_case: Use_Case, Test_data_set: Data_set, time_step: float, end_iteration: int, step_size_polish: int):
    """Function that makes extra training by seeing what the MPC would do in the states that the NN gets to the variable step_size_polish tells the program how often the mpc should be asked for"""
    extra_data_set = Data_set(len(use_case.labels_input), len(use_case.labels_output))

    for j in range(int(np.ceil(end_iteration/step_size_polish))):
        row = Test_data_set.give_input_row(step_size_polish*j)
        get_mpc_trace(use_case, row, step_size_polish*j, step_size_polish, time_step, extra_data_set)

    extra_data_set.delete_False_outputs()

    return extra_data_set

def get_mpc_trace(use_case, input, start_idx, step_size_polish, time_step, extra_data_set):
    for idx in range(step_size_polish):
        control_action = use_case.give_expert_action(input, start_idx+idx)
        extra_data_set.add_rows([input + control_action])
        input = use_case.give_next_state(input, control_action, time_step, idx)

def get_MSE(Test_data_set: Data_set, perfect_path: Data_set, MSE_result_lists: List[List[float]]):
    """Calculates the mean square error between the positions and the speeds the NN gets to and the perfect path that the MPC would take. This is saved in a list"""
    columns_test_data = Test_data_set.give_columns(give_input=True, give_output=False)
    columns_perfect_path = perfect_path.give_columns(give_input=True, give_output=False)

    for idx in range(Test_data_set.input_width):
        #solve bug of perfect path
        state_parameter_MSE = np.sum(np.power(np.array(columns_test_data[idx])-np.array(columns_perfect_path[idx][0:len(columns_test_data[idx])]),2))
        MSE_result_lists[idx].append(state_parameter_MSE)

def save_MSE(MSE_result_lists: List[List[float]], labels_input: List[str], naam: str):
    """Saves the MSE results and plot figures and also saves these"""
    with open(naam+'\MSE_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for idx, MSE_result_list in enumerate(MSE_result_lists):
            writer.writerow(MSE_result_list)
            _ = plt.subplot()
            plt.plot(MSE_result_list)
            plt.savefig(naam+'\\'+labels_input[idx]+'_MSE_fig')
            plt.clf()

def get_total_MSE_results(use_case: Use_Case, polish_iterations: int, map_name: str):
    """This function gets and saves the MSE results from each iteration"""
    MSE_total_results = give_list_list(len(use_case.labels_input))

    for polish_it in range(polish_iterations):
        current_map_name = map_name + '\iteration_' + str(polish_it)
        current_map_name=current_map_name+'\MSE_results\MSE_results.csv'
        with open(current_map_name) as file_name:
            MSE_results = np.loadtxt(file_name, delimiter=",")

        for idx in range(len(use_case.labels_input)):
            MSE_total_results[idx].append(sum(MSE_results[idx]))
    
    save_total_MSE_results(use_case, map_name, MSE_total_results)

def save_total_MSE_results(use_case: Use_Case, map_name: str, MSE_total_results: List[List[float]]):
    """This function saves the total MSE results"""
    with open(map_name+'\MSE_total_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for idx in range(len(use_case.labels_input)):
            writer.writerow(MSE_total_results[idx])
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
    pass
