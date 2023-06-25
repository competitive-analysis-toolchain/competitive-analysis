import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import csv
from typing import List
from Use_Case_class import Use_Case

def hypertune(use_case: Use_Case, normalizer: tf.keras.layers.experimental.preprocessing.Normalization, train_features: pd.DataFrame, train_labels: pd.DataFrame, training_epochs: int, training_factor: int, directory_name: str, name: str) -> kt.engine.hyperparameters.HyperParameters:
    """This functions hypertunes a model and gives back the best hyperparameters"""
    hyper_model = use_case.give_hypermodel(normalizer, train_features)

    tuner = kt.Hyperband(hyper_model,
                        objective='val_loss',
                        max_epochs=training_epochs,
                        factor=training_factor,
                        directory=directory_name,
                        project_name=name)

    tuner.search(train_features, train_labels, validation_split=0.2)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps

def save_hyperparametersearch_results(directory_name: str, best_hps: kt.engine.hyperparameters.HyperParameters, training_epochs: int, training_factor: int, hyperparameters: List[str]):
    """This functions saves the best hyperparameters and the settings of the hyperparametertuning"""
    outputfile= open(directory_name + '/hyper_outputfile.txt', "a")

    for idx in range(len(best_hps.space)):
        outputfile.write(f"{best_hps.space[idx]}\n")

    outputfile.write(f"The amount epochs are: {training_epochs}\n")
    outputfile.write(f"The training factor is: {training_factor}\n")
    outputfile.write(f"The best found results are\n")

    for hyperparameter in hyperparameters:
        outputfile.write(f"{hyperparameter}: {best_hps.get(hyperparameter)}\n")

    outputfile.close()

def get_hyperparameters_from_best_hps(use_case: Use_Case, best_hps: kt.engine.hyperparameters.HyperParameters) -> List:
    """This function gives back a list with the best hyperparameters from best_hps"""
    hyperparameters = []
    for hyperparameter_label in use_case.hyperparameters:
        hyperparameters.append(best_hps.get(hyperparameter_label))

    return hyperparameters

def save_hyperparameters_to_file(use_case: Use_Case, best_hps: kt.engine.hyperparameters.HyperParameters, file_name: str):
    """This function saves hyperparameters to a file"""
    outputfile= open(file_name, "w")
    for hyperparameter_label in use_case.hyperparameters:
        outputfile.write(str(best_hps.get(hyperparameter_label))+'\n')
    outputfile.close()
    
def get_hyperparameters_from_file(file_name: str) -> List:
    """This function gives back a list with the best hyperparameters from a file"""
    with open(file_name) as file:
        csvreader = csv.reader(file)
        hyperparameters = []
        for row in csvreader:
            if row[0][0].isdigit():
                hyperparameters.append(float(row[0]))
            else:
                hyperparameters.append(row[0])
    
    return hyperparameters

if __name__ == "__main__":
 
    print('hey')

    
