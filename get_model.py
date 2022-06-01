from re import L
from more_itertools import last
from model_manager import *
from data_manager import *
import pyuavcan_v0, time, math
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.activations import selu
import keras.backend as K
import numpy as np
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pickle

def linear(x):
    return x

def custom_selu(x, alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946):
    if x >= 0:
        return np.float32(np.float32(scale) * np.float32(x))
    else:
        return np.float32(np.float32(scale) * np.float32(alpha) * np.float32((np.float32(exp(x)) - 1)))


model = create_model_rnn(5)
set_of_weights = model.get_weights()

for w in set_of_weights:
    print(w.shape, w)


(train_Y, train_X), (test_Y, test_X) = rnn_dataset(n_historical = 5)
out_comparison = []
    
values_RNN_initial_state =  [0] * set_of_weights[0].shape[1]
n_inputs = set_of_weights[0].shape[0]
e = 0

for i in range(10):
    values_RNN_prev =   values_RNN_initial_state
    values_RNN =        [0] * set_of_weights[0].shape[1]
    # values_O =          [0] * 1
    # values_O =          [0] * set_of_weights[3].shape[1]
    
    for j in range(len(test_X[i])):
        for rnn_cell_index in range(len(values_RNN)):
            s = 0
            
            for index_I in range(n_inputs):
                s += np.float32(np.float32(test_X[i][j][index_I]) * np.float32((set_of_weights[0][index_I][rnn_cell_index])))
                
            for prev_rnn_cell_index in range(len(values_RNN)):
                s += np.float32(np.float32(values_RNN_prev[prev_rnn_cell_index]) * np.float32(set_of_weights[1][prev_rnn_cell_index][rnn_cell_index]))
                
            values_RNN[rnn_cell_index] = np.float32(np.float32(custom_selu(np.float32(s + np.float32(set_of_weights[2][rnn_cell_index])))))

            # for index_v_O in range(len(values_O)):
            #     s = 0
            #     for index_v_RNN in range(len(values_RNN)):
            #         s += np.float32(np.float32(values_RNN[index_v_RNN]) * np.float32(set_of_weights[3][index_v_RNN][index_v_O]))
            #     values_O[index_v_O] = np.float32(np.float32(s) + np.float32(set_of_weights[4][index_v_O]))
            
        values_RNN_prev = values_RNN
    
    # exit()
        
    pred_model_results = model.predict(np.array([test_X[i]]))
    for i in range(len(values_RNN)):
        e += abs(values_RNN[i] -  pred_model_results[0][i])
    
print(e)