import os
import csv
import shutil

from numpy import linspace
import numpy as np
from numpy import pi, cos, sin, tan, square
from Use_Case_class import Data_set
from general_methods import Hypertune_and_train_NN, Train_NN


def retrain_for_cegar(use_case, counter_example_file, amount_of_input_labels, amount_of_output_labels, amount_interval_points, amount_of_reference_points, saved_data, hyper_training_epochs, hyper_training_factor, NN_training_epochs, output_map, hypertune, hyperparameterfile):
    print(hypertune)
    counter_examples = load_counterexamples(counter_example_file, amount_of_input_labels + amount_of_output_labels)
    extra_datapoints = generate_extra_from_counterexamples(counter_examples[0], counter_examples[1], amount_of_input_labels + amount_of_output_labels, amount_interval_points)
    if amount_of_reference_points>0:
        extra_datapoints = add_reference_points_to_extra_datapoints(extra_datapoints, amount_of_reference_points)
    data_set = Data_set(amount_of_input_labels+amount_of_reference_points*2, amount_of_output_labels)
    #print(extra_datapoints)
    #print(data_set.input_width)
    #print(data_set.output_width)
    data_set.load_data_from_row_list(extra_datapoints)
    generate_control_action_for_datapoints(use_case, data_set)
    shutil.copyfile(saved_data, 'old_data_plus_extra_data.csv')
    data_set.save_data('old_data_plus_extra_data.csv')
    if hypertune:
        Hypertune_and_train_NN(use_case, 'old_data_plus_extra_data.csv', hyper_training_epochs, hyper_training_factor, NN_training_epochs, output_map, hyperparameterfile)
    else:
        Train_NN(use_case, hyperparameterfile, NN_training_epochs, output_map, saved_data = 'old_data_plus_extra_data.csv', processed_data = False)
    
def load_counterexamples(counter_example_file, amount_of_inputs):
    with open(counter_example_file) as file_name:
        csvreader = csv.reader(file_name)
        counter_examples_low = []
        #counter_examples_high = []
        for idx, row in enumerate(csvreader):
            if idx !=0:
                if row[0]=="new data":
                    counter_examples_low.append([float(i) for i in row[1:amount_of_inputs+1]])
                    #counter_examples_high.append([float(i) for i in row[amount_of_inputs+1:2*amount_of_inputs+1]])

    return counter_examples_low#, counter_examples_high

def generate_extra_from_counterexamples(counter_example_low, counter_example_high, amount_of_inputs, amount_interval_points):
    extra_datapoints = [[]]

    for jdx, counter_example_input_low in enumerate(counter_example_low):
        if jdx != amount_of_inputs-1:
            counter_example_input_high = counter_example_high[jdx]
            interval_points = list(linspace(counter_example_input_low, counter_example_input_high, num=amount_interval_points))

            new_extra_datapoints = []

            for data_point in extra_datapoints:
                for interval_point in interval_points:
                    new_data_point = data_point.copy()
                    new_data_point.append(interval_point)

                    new_extra_datapoints.append(new_data_point)
                
            extra_datapoints = new_extra_datapoints
   
    return extra_datapoints

def generate_control_action_for_datapoints(use_case, data_set):
    columns = data_set.give_columns(give_input=True, give_output=False)
    control_variables = use_case.give_expert_actions(columns)
    
    data_set.load_data_from_col_list(control_variables, load_only_output = True)
    data_set.delete_False_outputs()
    return data_set

def add_reference_points_to_extra_datapoints(extra_datapoints, amount_of_reference_points):
    """only for motion planning"""
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10

    new_extra_datapoints = []

    for datapoint in extra_datapoints:
        reference_points = get_reference_points(amount_of_reference_points, ref_path, datapoint[0], datapoint[1])
        new_datapoint = datapoint[0:3]+reference_points
        new_extra_datapoints.append(new_datapoint)
    
    return new_extra_datapoints

def get_reference_points(amount_of_reference_points, ref_path, x, y):
    close_index = find_closest_point([x, y], ref_path, 0)
    reference_points=[]

    for steps_foward in range(amount_of_reference_points):
        if close_index+steps_foward<31:
            x_dis = ref_path['x'][close_index+steps_foward]-x
            y_dis = ref_path['y'][close_index+steps_foward]-y
        else:
            x_dis = ref_path['x'][30]-x
            y_dis = ref_path['y'][30]-y
        reference_points.append(x_dis)
        reference_points.append(y_dis)
    
    return reference_points

def find_closest_point(pose, reference_path, start_index):
    # x and y distance from current position (pose) to every point in 
    # the reference path starting at a certain starting index
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]
    # Index of closest point by Pythagoras theorem
    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    #print('find_closest_point results in', index_closest)
    return index_closest
# if __name__ == "__main__":
#     print("h")
#     counter_examples_low = load_counterexamples("retraining_a.csv", 5)
#     print(counter_examples_low)
  
