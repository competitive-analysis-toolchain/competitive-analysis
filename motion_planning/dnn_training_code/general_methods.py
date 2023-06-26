import os
import string
import pandas as pd
import numpy as np
import csv
from typing import List

from general_NN_training_functions import get_and_process_data, get_normalizer, fit_model, plot_and_save_figures, save_model, write_output_file, get_total_error_results, get_test_results, load_NN_from_weights
from general_hyper_functions import hypertune, save_hyperparametersearch_results, get_hyperparameters_from_best_hps, save_hyperparameters_to_file, get_hyperparameters_from_file
from general_Test_NN_functions import Evaluate_NN_with_MPC, get_total_MSE_results
from Use_Case_class import Data_set

from Use_Case_class import Use_Case
#from Use_cases.Scara.scara_use_case import SCARA_Use_case
from Use_cases.Cart_pole.cart_pole_use_case import Cart_pole_Use_case
from Use_cases.Car_example.car_example_use_case import  Car_example_Use_case
from Use_cases.Motion_planning.motion_planning_use_case import Motion_planning_Use_case

"""This File contains higher abstract methods to hypertune and train Neural Networks"""

def polish(use_case: Use_Case, saved_data: str, hyper_training_epochs: int, hyper_training_factor: int, NN_training_epochs: int, polish_iterations: int, step_size_polish: int, folder_name: str, project_name: str):
    """This functions implements the Polish method"""

    current_dataset = saved_data
    map_name = folder_name + project_name

    for polish_it in range(polish_iterations):
        current_map_name = map_name + '/iteration_' + str(polish_it)
        Next_map_name = map_name + '/iteration_' + str(polish_it+1)
        os.makedirs(Next_map_name)
        os.makedirs(current_map_name+'/MSE_results')

        [train_features, test_features, train_labels, test_labels]= get_and_process_data(current_dataset, use_case.NN_labels_input, use_case.labels_output)
        save_test_dataset(test_features, test_labels, current_map_name + '/test_data_set_iteration_' + str(polish_it+1))

        Hypertune_NN(use_case, hyper_training_epochs, hyper_training_factor, current_map_name, current_map_name+'/hyperparameters', saved_data = False, processed_data = [train_features, test_features, train_labels, test_labels])
        Train_NN(use_case, current_map_name+'/hyperparameters', NN_training_epochs, current_map_name, saved_data = False, processed_data = [train_features, test_features, train_labels, test_labels])
        Evaluate_NN_with_MPC(use_case, current_map_name+'/output_NN_training/dnn_modelweigths.h5', current_map_name+'/hyperparameters', current_dataset, Next_map_name+'/data_set_iteration_'+str(polish_it+1), current_map_name+'/MSE_results', step_size_polish, clone_existing_data=False, generate_extra_data=True)

        current_dataset = Next_map_name+'/data_set_iteration_'+str(polish_it+1)
    
    get_total_MSE_results(use_case, polish_iterations, map_name)
    get_total_error_results(use_case, polish_iterations, map_name)


def Hypertune_and_train_NN(use_case: Use_Case, saved_data: string, hyper_training_epochs: int, hyper_training_factor: int, NN_training_epochs: int, output_map: string, hyperparameter_file: bool = False):
    """Hypertune and train a NN"""

    [train_features, test_features, train_labels, test_labels] = get_and_process_data(saved_data, use_case.NN_labels_input, use_case.labels_output)
    Hypertune_NN(use_case, hyper_training_epochs, hyper_training_factor, output_map, hyperparameter_file, saved_data = False, processed_data = [train_features, test_features, train_labels, test_labels])
    Train_NN(use_case, hyperparameter_file, NN_training_epochs, output_map, saved_data = False, processed_data = [train_features, test_features, train_labels, test_labels])

def Hypertune_NN(use_case: Use_Case, hyper_training_epochs: int, hyper_training_factor: int, output_map: str, hyperparameter_file: str, saved_data: str = False, processed_data: List[pd.DataFrame] = False):
    """Hypertune a NN"""
    
    if saved_data:
        [train_features, test_features, train_labels, test_labels] = get_and_process_data(saved_data, use_case.NN_labels_input, use_case.labels_output)
    elif processed_data:
        [train_features, test_features, train_labels, test_labels] = processed_data
    else:
        print('Error: No data was given')

    normalizer = get_normalizer(train_features)
    best_hps = hypertune(use_case, normalizer, train_features, train_labels, hyper_training_epochs, hyper_training_factor, output_map, 'hyper_trials')

    save_hyperparametersearch_results(output_map, best_hps, hyper_training_epochs, hyper_training_factor, use_case.hyperparameters)
    save_hyperparameters_to_file(use_case, best_hps, hyperparameter_file)


def Train_NN(use_case: Use_Case, hyperparameter_file: str, NN_training_epochs: int, output_map: str, saved_data: str = False, processed_data: List[pd.DataFrame] = False):
    """Train a NN"""

    output_NN_map_name = output_map + '/output_NN_training'
    os.makedirs(output_NN_map_name)
    
    if saved_data:
        [train_features, test_features, train_labels, test_labels] = get_and_process_data(saved_data, use_case.NN_labels_input, use_case.labels_output)
    elif processed_data:
        [train_features, test_features, train_labels, test_labels] = processed_data
    else:
        print('Error: No data was given')

    save_test_dataset(test_features, test_labels, 'test_data_set_iteration.csv')

    normalizer = False#get_normalizer(train_features)
    hyper_parameters = get_hyperparameters_from_file(hyperparameter_file)
    dnn_model = use_case.give_NN_model(normalizer, hyper_parameters, train_features)
    history = fit_model(dnn_model, train_features, train_labels, NN_training_epochs)

    plot_and_save_figures(history, dnn_model, test_features, test_labels, use_case.labels_output, output_NN_map_name)
    test_results = get_test_results(dnn_model, test_features, test_labels)
    save_model(use_case, dnn_model, output_NN_map_name)
    write_output_file(dnn_model, output_NN_map_name, hyper_parameters, use_case.hyperparameters, NN_training_epochs, test_results)

def Retrain_NN(use_case: Use_Case, hyperparameter_file: str, saved_weights: str, NN_training_epochs: int, output_map: str, saved_data: str = False, processed_data: List[pd.DataFrame] = False):
    """Retrain a NN"""

    output_NN_map_name = output_map + '/output_NN_retraining'
    os.makedirs(output_NN_map_name)
    
    if saved_data:
        [train_features, test_features, train_labels, test_labels] = get_and_process_data(saved_data, use_case.NN_labels_input, use_case.labels_output)
    elif processed_data:
        [train_features, test_features, train_labels, test_labels] = processed_data
    else:
        print('Error: No data was given')

    save_test_dataset(test_features, test_labels, 'test_data_set_iteration.csv')

    hyper_parameters = get_hyperparameters_from_file(hyperparameter_file)
    saved_model= load_NN_from_weights(use_case, hyper_parameters, train_features, saved_weights)

    history = fit_model(saved_model, train_features, train_labels, NN_training_epochs)

    plot_and_save_figures(history, saved_model, test_features, test_labels, use_case.labels_output, output_NN_map_name)
    test_results = get_test_results(saved_model, test_features, test_labels)
    save_model(use_case, saved_model, output_NN_map_name, '')
    write_output_file(saved_model, output_NN_map_name, '', hyper_parameters, use_case.hyperparameters, NN_training_epochs, test_results)

def save_test_dataset(test_features, test_labels, save_place):
    #works only for the cart pole use case
    test_dataset_list = test_features.copy()
    test_labels_list = test_labels.copy()

    test_dataset_list = test_dataset_list.values.tolist()
    test_labels_list = test_labels_list.values.tolist()
    for idx, test_label in enumerate(test_labels_list):
        for label in test_label:
            test_dataset_list[idx].append(label)

    test_dataset = Data_set(len(test_features.columns),len(test_labels.columns))
    test_dataset.load_data_from_row_list(test_dataset_list)
    test_dataset.save_data(save_place)

if __name__ == "__main__":
    use_case = Cart_pole_Use_case()
    use_case.set_self_parameters()

    # polish(use_case, 'Use_cases\Car_example\MPC_data\data_set_car_example.csv', 2, 3, 4, 3, 'Dagger_results\car_example\\', 'experiment_delete')
    polish(use_case, 'Use_cases/Cart_pole/MPC_data/data_set_mpc_example.csv', 2, 3, 4, 2, 3, 'Polish_results/cart_pole/', 'experiment_delete')
    # polish(use_case, 'Use_cases\Motion_planning\MPC_data\data_set_motion_planning_MPC_3_refp.csv', 2, 3, 4, 3, 'Dagger_results\motion_planning\\', 'experiment_delete')
    # polish(use_case, 'Use_cases\Scara\MPC_data\data_set_scara.csv', 20, 3, 40, 3, 'Dagger_results\scara\\', 'experiment_delete')
    
