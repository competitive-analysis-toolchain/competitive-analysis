import numpy as np
import matplotlib.pyplot as plt
import csv
from general_Test_NN_functions import get_control_action_from_NN
from general_hyper_functions import get_hyperparameters_from_file
from general_NN_training_functions import get_normalizer, get_and_process_data, load_NN_from_weights
from typing import List

from Use_cases.Scara.scara_use_case import SCARA_Use_case
from Use_cases.Cart_pole.cart_pole_use_case import Cart_pole_Use_case
from Use_cases.Car_example.car_example_use_case import  Car_example_Use_case
from Use_cases.Motion_planning.motion_planning_use_case import Motion_planning_Use_case
from Use_Case_class import Use_Case, give_list_list

def make_error_figures():
    """This function makes figures of the error results"""
    map_name = 'Polish_results/cart_pole/experiment_'
    for i in [1]:
        current_map_name = map_name + str(i) + '/error_total_results.csv'
        with open(current_map_name) as file_name:
            MSE_total_results = np.loadtxt(file_name, delimiter=",")
        plt.figure(1)
        plt.plot(MSE_total_results[0], label='U1')
        plt.xlabel('Polish Iteration')
        plt.ylabel('mean of error')
        plt.legend()

        plt.figure(2)
        plt.plot(MSE_total_results[1], label='U2')
        plt.xlabel('Polish Iteration')
        plt.ylabel('meddian of error')
        plt.legend()

        plt.figure(3)
        plt.plot(MSE_total_results[2], label='U1')
        plt.xlabel('Polish Iteration')
        plt.ylabel('standard deviation of error')
        plt.legend()

   

    plt.show(block=True)   

def make_MSE_figures():
    """This function makes figures of the MSE results"""
    map_name = 'Dagger_results/cart_pole/experiment_'
    label_names = [0,50,100,200]
    extra_names = ['','_loss','_restricted']
    start_array = 0
    for extra_name in extra_names:
        for i in [1,2,3]:
            current_map_name = map_name + str(i) + extra_name + '/MSE_total_results.csv'
            with open(current_map_name) as file_name:
                MSE_total_results = np.loadtxt(file_name, delimiter=",")
            plt.figure(1)
            plt.plot(MSE_total_results[0][start_array:], label = 'hyper_epochs_'+str(label_names[i])+extra_name)
            plt.xlabel('Dagger Iteration')
            plt.ylabel('Sum of sum of square errors of pos')
            plt.legend()

            plt.figure(2)
            plt.plot(MSE_total_results[1][start_array:], label = 'hyper_epochs_'+str(label_names[i])+extra_name)
            plt.xlabel('Dagger Iteration')
            plt.ylabel('Sum of sum of square errors of theta')
            plt.legend()

            plt.figure(3)
            plt.plot(MSE_total_results[2][start_array:], label = 'hyper_epochs_'+str(label_names[i])+extra_name)
            plt.xlabel('Dagger Iteration')
            plt.ylabel('Sum of sum of square errors of dpos')
            plt.legend()

            plt.figure(4)
            plt.plot(MSE_total_results[3][start_array:], label = 'hyper_epochs_'+str(label_names[i])+extra_name)
            plt.xlabel('Dagger Iteration')
            plt.ylabel('Sum of sum of square errors of dtheta')
            plt.legend()

    plt.show(block=True)   

def check_NN_output(use_case: Use_Case, saved_data: str, hyperparameter_file: str, weights_file: str):
    """This function test a NN and saves all the control actions that fall outside of min_edge and max_edge"""
    min_edge = [[-2]]
    max_edge = [[2]]

    [train_features, _, _, _] = get_and_process_data(saved_data, use_case.NN_labels_input, use_case.labels_output, 1.0)
    hyper_parameters = get_hyperparameters_from_file(hyperparameter_file)
    NN = load_NN_from_weights(use_case, hyper_parameters, train_features, weights_file)
    
    data = get_data_from_file(saved_data)
    control_actions = give_list_list(len(use_case.labels_output))

    for row in data:
        input = row[:len(use_case.labels_input)]
        control_action = get_control_action_from_NN(use_case, NN, input)

        for idx, control_par in enumerate(control_action):
            if control_par < min_edge[idx] or control_par > max_edge[idx]:
                control_actions[idx].append(control_par)

    save_bad_points(control_actions)

def get_data_from_file(saved_data: str) -> List[List[float]]:
    """This function gets data from a file"""
    with open(saved_data) as file_name:
        csvreader = csv.reader(file_name)
        data = []
        for row in csvreader:
            data.append([float(i) for i in row])
    
    return data

def save_bad_points(control_actions: List[List[float]]):
    """This function saves the bad control actions"""
    with open('bad_points.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for control_action_list in control_actions:
            writer.writerow(control_action_list)

if __name__ == "__main__":
    # make_error_figures()
    make_MSE_figures()
    # use_case =  Cart_pole_Use_case()
    # use_case.set_self_parameters()

    # check_NN_output(use_case, 'Use_cases\Cart_pole\MPC_data\data_set_mpc_example.csv', 'Dagger_results\cart_pole\experiment_delete\iteration_0\hyperparameters', 'Dagger_results\cart_pole\experiment_delete\iteration_0\output_NN_training\dnn_modelweigths.h5')
