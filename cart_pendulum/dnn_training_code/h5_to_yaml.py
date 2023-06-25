#!/usr/bin/python
import tensorflow as tf

#from keras.models import Sequential
from tensorflow.keras.models import Sequential
#from keras import models
from tensorflow.keras import models
import numpy as np
import sys
import yaml

def h5_to_yml(argv, custom_objects_use_case=None):
    input_filename = argv[0]
    output_filename = argv[1]

    model = models.load_model(input_filename, custom_objects=custom_objects_use_case)

    dnn_dict = {}
    dnn_dict['weights'] = {}
    dnn_dict['offsets'] = {}
    dnn_dict['activations'] = {}

    layer_count = 1
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            dnn_dict['weights'][layer_count] = []
            for row in layer.get_weights()[0].T:
                a = []
                try:    
                    for column in row: a.append(float(column))
                except:
                    a.append(float(row))
                dnn_dict['weights'][layer_count].append(a)
            
            dnn_dict['offsets'][layer_count] = []
            for row in layer.get_weights()[1].T:
                dnn_dict['offsets'][layer_count].append(float(row))
            
            if 'normalization' not in str(layer.output):
                if hasattr(layer, 'activation'):
                    if 'sigmoid' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Sigmoid'
                    elif 'tanh' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Tanh'
                    elif 'relu' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Relu'
                    elif 'swish' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Swish'
                    else:
                        dnn_dict['activations'][layer_count] = 'Linear'
                else:
                    dnn_dict['activations'][layer_count] = 'Sigmoid'
            else:
                dnn_dict['activations'][layer_count] = 'Linear'
            layer_count += 1

    with open(output_filename, 'w') as f:
        yaml.dump(dnn_dict, f)

if __name__ == '__main__':
    h5_to_yml(sys.argv[1:])
