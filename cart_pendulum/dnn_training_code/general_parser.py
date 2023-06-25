import os
import csv
import shutil

from numpy import linspace
import numpy as np
from Use_Case_class import Data_set
from general_methods import Hypertune_and_train_NN, Train_NN


def retrain_for_cegar(use_case, counter_example_file, amount_of_input_labels, amount_of_output_labels, amount_interval_points, saved_data, hyper_training_epochs, hyper_training_factor, NN_training_epochs, output_map, hypertune, hyperparameterfile):
    print(hypertune)
    counter_examples = load_counterexamples(counter_example_file, amount_of_input_labels + amount_of_output_labels)
    extra_datapoints = generate_extra_from_counterexamples(counter_examples[0], counter_examples[1], amount_of_input_labels + amount_of_output_labels, amount_interval_points)
    data_set = Data_set(amount_of_input_labels, amount_of_output_labels)
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


# if __name__ == "__main__":
#     print("h")
#     counter_examples_low = load_counterexamples("retraining_a.csv", 5)
#     print(counter_examples_low)
  