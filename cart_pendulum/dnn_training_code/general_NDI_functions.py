import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List
from Use_Case_class import Data_set
from general_Test_NN_functions import get_control_action_from_NN
from general_hyper_functions import get_hyperparameters_from_file
from general_NN_training_functions import get_and_process_data, load_NN_from_weights
from general_methods import Hypertune_and_train_NN

from Use_Case_class import Use_Case
#from Use_cases.Scara.scara_use_case import SCARA_Use_case
from Use_cases.Cart_pole.cart_pole_use_case import Cart_pole_Use_case
from Use_cases.Car_example.car_example_use_case import  Car_example_Use_case
from Use_cases.Motion_planning.motion_planning_use_case import Motion_planning_Use_case

DIFF_BETWEEN_CONTROL_ACTIONS = 0.01

def NDI(use_case: Use_Case, iter_max: int, k: int, hyper_training_epochs: int, hyper_training_factor: int, NN_training_epochs: int, folder_name: str, project_name: str):
    """This function implements non divergent imitation"""
    paths = []
    end_iterations_list = []
    path_lengths_mean = []
    path_lengths_std = []
    NN = False

    for id in range(iter_max):
        current_map_name = folder_name + project_name + '/iteration_' + str(id)
        os.makedirs(current_map_name)

        training_data_set = Data_set(len(use_case.labels_input), len(use_case.labels_output))
        end_iterations_list = get_k_path_starts(k, use_case, paths, end_iterations_list)
        new_paths = get_new_paths(use_case, training_data_set, paths, NN, end_iterations_list, id)
        
        NN = get_verifiable_policy(use_case, training_data_set, current_map_name, hyper_training_epochs, hyper_training_factor, NN_training_epochs)

        paths = new_paths

        if id != 0:
            get_path_lengths(new_paths, k, path_lengths_mean, path_lengths_std)
    
    save_mean_and_std(path_lengths_mean, path_lengths_std, folder_name + project_name)

def get_k_path_starts(k: int, use_case: Use_Case, paths: List[List[List[float]]], end_iterations_list: List[int]) -> List[int]:
    """This function samples k new start points"""
    start_points, end_iterations = use_case.sample_k_start_points(k)
    end_iterations_list = end_iterations_list + end_iterations

    start_points_data_set = Data_set(len(use_case.labels_input), len(use_case.labels_output))
    start_points_data_set.load_data_from_row_list(start_points)

    columns = start_points_data_set.give_columns(give_input=True, give_output=False)
    control_variables = use_case.give_expert_actions(columns)
    
    start_points_data_set.load_data_from_col_list(control_variables, load_only_output = True)
    start_points_data_set.delete_False_outputs()

    start_points_with_control_actions = start_points_data_set.give_rows(give_input=True, give_output=True)

    for start_point_with_control_actions in start_points_with_control_actions:
        paths.append([start_point_with_control_actions])
    
    return end_iterations_list

def get_new_paths(use_case: Use_Case, training_data_set: Data_set, paths: List[List[List[float]]], NN: tf.keras.Sequential, end_iterations_list: List[int], id: int):
    """This function Extends the paths and saves them in the right lists"""
    new_paths = []

    for jd, path in enumerate(paths):
        new_path = ExteND(use_case, path, NN, id, end_iterations_list[jd])

        for path_point in new_path:
            training_data_set.add_rows([path_point])
        
        if len(new_path) > len(path):
            new_paths.append(new_path)
        else:
            new_paths.append(path)
    
    training_data_set.delete_False_outputs()

    return new_paths

def ExteND(use_case: Use_Case, path: List[List[float]], NN: tf.keras.Sequential, id: int, endindex: int) -> List[List[float]]:
    """This function Extends the paths until the control action of the NN is to different from the MPC"""
    new_path = []

    for idx in range(endindex):
        if idx+1 > len(path):
            sim_input = new_path[-1][:len(use_case.labels_input)]
            sim_output = new_path[-1][-len(use_case.labels_output):]
            
            next_state = use_case.give_next_state(sim_input, sim_output, use_case.timesteps[0], idx)
            control_action = use_case.give_expert_action(next_state+[idx])

            path_extension = next_state + control_action
        else:
            path_extension = path[idx]
        
        new_path.append(path_extension)

        if id != 0:
            break_par = check_NN_output(path_extension, NN)

            if break_par:
                break
                    
    return new_path

def check_NN_output(path_extension: List[float], NN: tf.keras.Sequential) -> bool:
    """checks if the control action from the NN is not to different from the MPC"""
    control_input = path_extension[:len(use_case.labels_input)]

    NN_control_action = get_control_action_from_NN(use_case, NN, control_input)
    MPC_control_action = path_extension[-len(use_case.labels_output):]

    break_par = False

    for jdx, NN_control in enumerate(NN_control_action):
        if np.abs(MPC_control_action[jdx]-NN_control) > DIFF_BETWEEN_CONTROL_ACTIONS:
            break_par = True

    return break_par

'''def get_verifiable_policy(use_case: Use_Case, training_data_set: Data_set, current_map_name: str, hyper_training_epochs: int, hyper_training_factor: int, NN_training_epochs: int) -> tf.keras.Sequential:
    """hypertune and train a NN and give it back"""
    NN_training_data_set = use_case.give_NN_data(training_data_set)
    NN_training_data_set.save_data(current_map_name + '\\training_data.csv')
    Hypertune_and_train_NN(use_case, current_map_name + '\\training_data.csv', hyper_training_epochs, hyper_training_factor, NN_training_epochs, current_map_name, current_map_name+'\hyperparameters')

    [train_features, test_features, train_labels, test_labels] = get_and_process_data(current_map_name + '\\training_data.csv', use_case.NN_labels_input, use_case.labels_output)
    hyper_parameters = get_hyperparameters_from_file(current_map_name + '\\hyperparameters')
    NN = load_NN_from_weights(use_case, hyper_parameters, train_features, current_map_name + '\output_NN_training\dnn_modelweigths.h5') 
    
    return NN'''
def get_verifiable_policy(use_case: Use_Case, training_data_set: Data_set, current_map_name: str, hyper_training_epochs: int, hyper_training_factor: int, NN_training_epochs: int) -> tf.keras.Sequential:
    """Hypertune and train a neural network and return it"""

    # Save training data
    training_data_path = current_map_name + '/training_data.csv'
    NN_training_data_set = use_case.give_NN_data(training_data_set)
    NN_training_data_set.save_data(training_data_path)

    # Hypertune and train the neural network
    hyperparameter_file = current_map_name + '/hyperparameters'
    Hypertune_and_train_NN(use_case, training_data_path, hyper_training_epochs, hyper_training_factor, NN_training_epochs, current_map_name, hyperparameter_file)

    # Get and process data for training
    train_features, test_features, train_labels, test_labels = get_and_process_data(training_data_path, use_case.NN_labels_input, use_case.labels_output)

    # Load neural network from weights
    weights_path = current_map_name + '/output_NN_training/dnn_modelweights.h5'
    hyper_parameters = get_hyperparameters_from_file(hyperparameter_file)
    NN = load_NN_from_weights(use_case, hyper_parameters, train_features, weights_path)

    return NN

def get_path_lengths(new_paths: List[List[List[float]]], k: int, path_lengths_mean: List[float], path_lengths_std: List[float]):
    """get the path lenghts from all the paths that are not the first k"""
    path_lengths = []

    for next_path in new_paths[k:]:
        path_lengths.append(len(next_path))
        
        path_lengths_mean.append(np.mean(path_lengths))
        path_lengths_std.append(np.std(path_lengths))

def save_mean_and_std(path_lengths_mean: List[float], path_lengths_std: List[float], file: str):
    """saves the mean and std of the path lengths"""
    with open(file+'/mean_and_std_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(path_lengths_mean)
        writer.writerow(path_lengths_std)

    _ = plt.subplot()
    plt.plot(path_lengths_mean)
    plt.savefig(file+'/path_length_mean_fig')
    plt.clf()

    _ = plt.subplot()
    plt.plot(path_lengths_std)
    plt.savefig(file+'/path_length_std_fig')
    plt.clf()    

if __name__ == "__main__":
    use_case =  Car_example_Use_case()
    use_case.set_self_parameters()
    NDI(use_case, 3, 20, 2, 3, 4, 'NDI_results/mpc_example/', 'experiment_delete')
    
