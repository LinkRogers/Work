import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

import multiprocessing
from multiprocessing import Pool
from functools import partial
#%matplotlib inline

#import im2video
#import data2video

#__all__ = ['im2video', 'data2video']

def clear_line():
    print(' '*80,end='\r')
    
#function to get activation of input data into a list
def get_activation(data, get_layer_output_functions): 
    layer_output = []
    for f in get_layer_output_functions[:-1]:
        layer_output.append(f(data.reshape(1,2))[0])

    for i in range(len(layer_output)):
        #layer_nonzero_index = np.flatnonzero(layer_output[i])
        layer_output[i][np.flatnonzero(layer_output[i])] = 1
        layer_output[i] = layer_output[i].astype(int)
        
    act_list = layer_output
    return act_list

#assign a different color to each different activation,use color index
def assign_color(activations,act_index):
    for i in range(len(activations[0])):
        binary = ''.join(str(e) for e in activations[act_index][i].astype(int))
    return int(binary,2)

def prepare_model(hid_layer_units, n_category):
    model = tf.keras.Sequential()
    for i in range(len(hid_layer_units)):
        if i == 0:
            model.add(Dense(hid_layer_units[i], input_shape=(2,), activation='relu',bias_initializer='random_uniform'))
        else:
            model.add(Dense(hid_layer_units[i], activation='relu',bias_initializer='random_uniform'))
    model.add(Dense(n_category, activation='softmax',bias_initializer='random_uniform'))
    # Configure a model for categorical classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model

def data_grid(grid, square_len):
    k = square_len
    x = np.linspace(-k,k,grid+1)
    y = np.linspace(-k,k,grid+1) 

    xv,yv = np.meshgrid( x , y )
    data = np.array([xv.flatten(),yv.flatten()])
    data = data.transpose()
    return data

def get_color(data):
    interval = int(np.ceil(255/len(data)**(1/3.0)))
    R = range(0,255,interval)
    G = range(0,255,interval)
    B = range(0,255,interval)
    rv,gv,bv = np.meshgrid(R,G,B)
    colors = np.array([rv.flatten(),gv.flatten(),bv.flatten()]).transpose()
    np.random.shuffle(colors)
    return colors

def get_data_activation(all_data, get_layer_output_functions):
    #loop for all data points to get activations
    i_data = 0
    activations = []

    for data in all_data:
        #act_progress = "get activation progress:{0}%".format(round((i_data + 1) * 100 / len(X_test)))
        #print(act_progress, end='\r')
        activations.append(get_activation(data,get_layer_output_functions))
        i_data = i_data + 1
        
    return activations

def get_data_color(activations,colors):
    f = partial(assign_color,activations)
    i_color=0
    data_activation_color=[]
    pool = Pool(8)
    
    for y in pool.map(f, range(len(activations))):
        i_color+=1
        #color_progress = "get data color progress:{0}% ".format(round((i_color + 1) * 100 / len(activations)))
        #print(color_progress, end='\r')
        data_activation_color.append(y)

    data_activation_color = np.asarray(data_activation_color)
    data_color_cluster = np.unique(data_activation_color)
    for color_i in range(len(data_activation_color)):
        [data_activation_color[color_i]] = np.where(data_color_cluster == data_activation_color[color_i])[0]
    data_activation_color = data_activation_color.astype(int)
    data_activation_color = colors[data_activation_color]/255
    pool.close()
    pool.join()
    
    return data_activation_color
