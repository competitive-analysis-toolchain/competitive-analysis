import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5_to_yaml import h5_to_yml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from typing import List
from Use_Case_class import Use_Case, give_list_list
from typing import Dict, List, Any

def get_and_process_data(saved_data: str, NN_labels_input: List[str], labels_output: List[str], train_and_test_frac: float = 0.8) -> List[pd.DataFrame]:
    """This function process data for training and hypertuning a NN"""
    """Get the data"""
    
    column_names = NN_labels_input + labels_output
    raw_dataset = pd.read_csv(saved_data, names=column_names)

    dataset = raw_dataset.copy()

    """Split the data into train and test"""

    train_dataset = dataset.sample(frac=train_and_test_frac, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    """Split features from labels"""

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = pd.DataFrame()
    test_labels = pd.DataFrame()
    for output_label in labels_output:
        train_labels[output_label]=train_features.pop(output_label)

        test_labels[output_label]=test_features.pop(output_label)

    return [train_features, test_features, train_labels, test_labels]

def get_normalizer(train_features: pd.DataFrame) -> tf.keras.layers.experimental.preprocessing.Normalization:
    """This function gives the normalizer"""
    print(np.array(train_features))
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    return normalizer

def fit_model(dnn_model: tf.keras.Sequential, train_features: pd.DataFrame, train_labels: pd.DataFrame, Trained_epochs: int) -> keras.callbacks.History:
    """This function fit a model"""
    # train_features=np.asarray(train_features).astype(np.float32)
    # train_labels=np.asarray(train_labels).astype(np.float32)
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=2, epochs=Trained_epochs)
    return history

#def get_test_results(dnn_model: tf.keras.Sequential, test_features: pd.DataFrame, test_labels: pd.DataFrame) -> dict[float]:
def get_test_results(dnn_model: tf.keras.Sequential, test_features: pd.DataFrame, test_labels: pd.DataFrame) -> Dict[float, Any]:
    """This function gets the test results from a trained NN"""
    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, batch_size=32 ,verbose=0)
    return test_results

def plot_and_save_figures(history: keras.callbacks.History, dnn_model: tf.keras.Sequential, test_features: pd.DataFrame, test_labels: pd.DataFrame, labels_output: List[str], map_name: str):
    """This function makes and saves figures and metrics from the NN"""
    
    plot_loss(history, map_name)
    test_predictions = dnn_model.predict(test_features)

    error_results_list = []

    for idx, output_label in enumerate(labels_output):
        
        test_labels_output, test_predictions_output = plot_predict_vs_real(test_labels, test_predictions, idx, output_label, map_name, len(labels_output) == 1)
        error_mean, error_median, error_std = plot_histogram(test_labels_output, test_predictions_output, output_label, map_name)
        
        error_results_list.append(error_mean)
        error_results_list.append(error_median)
        error_results_list.append(error_std)

    save_error_metrics(error_results_list, map_name)

def plot_loss(history: keras.callbacks.History, map_name: str):
    """This function makes and saves the figure with the loss and validation loss"""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error  [U1,U2]')
    plt.legend()
    plt.grid(True)
    plt.savefig(map_name + '/Figure_loss_and_val_loss_')
    plt.clf()

def plot_predict_vs_real(test_labels: pd.DataFrame, test_predictions: pd.DataFrame, idx: int, output_label: str, map_name: str, only_one_output: bool) -> List[float]:
    """This function makes and saves the figure that plots the real vs predicted results"""
    test_labels_output = []
    for i in range(len(test_labels.values)):
        test_labels_output.append(test_labels.values[i][idx])
    
    test_predictions_output = []
    if only_one_output:
        for i in range(len(test_predictions)):
            test_predictions_output.append(test_predictions[i][0])
    else:
        for i in range(len(test_predictions)):
            test_predictions_output.append(test_predictions[i][idx])

    _ = plt.axes(aspect='equal')
    plt.scatter(test_labels_output, test_predictions_output)
    plt.xlabel('True Values ' + output_label)
    plt.ylabel('Predictions ' + output_label)
    plt.savefig(map_name + '/predict_vs_real_' + output_label )
    plt.clf()

    return test_labels_output, test_predictions_output

def plot_histogram(test_labels_output: List[float], test_predictions_output: List[float], output_label: str, map_name: str) -> List[float]:
    """This function makes and saves the histogram with the faults between the real and the predicted results"""
    list_test_predictions_output=[]
    list_test_labels_output=[]

    for i in range(len(test_predictions_output)):
        list_test_predictions_output.append(test_predictions_output[i])
        list_test_labels_output.append(test_labels_output[i])

    error_U = np.array(list_test_predictions_output) - np.array(list_test_labels_output)
    plt.hist(error_U, bins=25)
    plt.xlabel('Prediction Error ' + output_label)
    _ = plt.ylabel('Count')
    plt.savefig(map_name + '/histogram_' + output_label )
    plt.clf()

    return [np.mean(error_U), np.median(error_U), np.std(error_U)]

def save_error_metrics(error_results_list: List[float], folder: str):
    """This function saves the error metrics in a file"""
    with open(folder+'/Error_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(error_results_list)

def save_model(use_case: Use_Case, dnn_model: tf.keras.Sequential, map_name: str):
    """This function saves the model, the weights and saves it as a yaml."""
    dnn_model.save(map_name+'/dnn_model')
    dnn_model.save_weights(map_name+'/dnn_model' + 'weigths.h5')
    h5_to_yml([map_name+'/dnn_model', map_name + '/dnn_model' +'_yaml'], use_case.custom_objects)

#def write_output_file(dnn_model: tf.keras.Sequential, map_name: str, hyperparameters: List, hyperparameters_labels: List[str], Trained_epochs: int, test_results: dict[float]):
def write_output_file(dnn_model: tf.keras.Sequential, map_name: str, hyperparameters: List, hyperparameters_labels: List[str], Trained_epochs: int, test_results: Dict[float, Any]):
    """This function saves the parameters of NN training and the test results"""
    #outputfile= open(map_name + '\\' + '_outputfile.txt', "a")
    outputfile = open(map_name + '/' + '_outputfile.txt', "a")
    for idx, hyperparameter_label in enumerate(hyperparameters_labels):
        outputfile.write(f"{hyperparameter_label}: {hyperparameters[idx]}\n")
    outputfile.write(f"The amount of trained epochs are: {Trained_epochs}\n")
    outputfile.write(f"The test loss is: {test_results['dnn_model']}\n")
    dnn_model.summary(print_fn=lambda x: outputfile.write(x + '\n'))
    outputfile.close()

def get_total_error_results(use_case: Use_Case, dagger_iterations: int, map_name: str):
    """This function gets the error results for each iteration and saves it in a file"""
    error_total_results = give_list_list(len(use_case.labels_output)*3)
    
    for dagger_it in range(dagger_iterations):
        current_map_name = map_name + '/iteration_' + str(dagger_it)
        current_map_name=current_map_name+'/output_NN_training/Error_results.csv'
        with open(current_map_name) as file_name:
            error_results = np.loadtxt(file_name, delimiter=",")

        for idx in range(len(use_case.labels_output)*3):
            error_total_results[idx].append(error_results[idx])

    save_total_error_results(use_case, error_total_results, map_name)

def save_total_error_results(use_case: Use_Case, error_total_results: List[List[float]], map_name: str):
    """This function saves the total error results in a file"""
    with open(map_name+'/Error_total_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for idx in range(len(use_case.labels_output)*3):
            writer.writerow(error_total_results[idx])

def load_NN_from_weights(use_case: Use_Case, hyper_parameters: List, train_features: pd.DataFrame, weights_file: str) -> tf.keras.Sequential:
    """This function loads a NN from saved weights"""
    normalizer = get_normalizer(train_features)
    NN = use_case.give_NN_model(normalizer, hyper_parameters, train_features)
    NN.load_weights(weights_file)
    return NN

if __name__ == "__main__":
    pass
    
