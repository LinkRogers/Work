import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

import multiprocessing
from multiprocessing import Pool
from functools import partial

#import im2video
#import data2video

self = ['clear_line', 'get_activation','assign_color','prepare_model','data_grid',\
       'get_color','get_data_activation','get_data_color','get_activation_prediction_transition']
sub_module = []

__all__ = self+sub_module

batch_size = 32

def clear_line():
    print(' '*80,end='\r')
    
def get_layer_output_functions(model):
    layer_output_functions = []
    for layer_i in range(len(model.layers)):
        layer_output_functions.append(K.function(model.layers[0].input,model.layers[layer_i].output))
    return layer_output_functions

#function to get activation of input data into a list
def get_activation(data, layer_output_functions): 
    layer_output = []
    for f in layer_output_functions[:-1]:
        layer_output.append(f(data.reshape(1,2))[0])

    for i in range(len(layer_output)):
        #layer_nonzero_index = np.flatnonzero(layer_output[i])
        layer_output[i][np.flatnonzero(layer_output[i])] = 1
        layer_output[i] = layer_output[i].astype(int)
        
    act_list = layer_output
    return act_list

#assign a different color to each different activation,use color index
def assign_color(activations,act_index):
    result = 0
    for i in range(len(activations[0])):
        binary = ''.join(str(e) for e in activations[act_index][i].astype(int))
        result += int(binary,2)
    return result

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

def get_data_activation(all_data, layer_output_functions):
    #loop for all data points to get activations
    i_data = 0
    activations = []

    for data in all_data:
        #act_progress = "get activation progress:{0}%".format(round((i_data + 1) * 100 / len(X_test)))
        #print(act_progress, end='\r')
        activations.append(get_activation(data,layer_output_functions))
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

def get_activation_prediction_transition(result_dir,num_epoch,X_test,model_data_labels): 
    model, data, labels = model_data_labels 
    layer_output_functions = get_layer_output_functions(model)
    for epoch_i in range(num_epoch):        
        activations = get_data_activation(X_test, layer_output_functions)
        #data_activation_color = get_data_color(activations,colors)
        Y_test = model.predict(X_test)

        np.savez_compressed(result_dir+'/'+'epoch'+str(format(epoch_i,'0>3')),\
                            activations = activations,\
			    prediction = Y_test)

        step = epoch_i+1
        history = model.fit(data, labels, epochs=1,verbose=0,steps_per_epoch=step,batch_size=batch_size)
        train_info = 'epoch '+ str(epoch_i+1)+': '+str(history.history)
        print(train_info, end='\n')
        
        root,model_name = os.path.split(result_dir)
        trainlog_path = root+'/trainlog'
        if not os.path.exists(trainlog_path):
            os.makedirs(trainlog_path)
        File = open(trainlog_path+'/'+model_name+'.txt','a+') 
        File.write(train_info+'\n')
        File.flush()
        File.close() 

def plot_data(data_path,X_test,colors):
    files = os.listdir(data_path)
    files = [file for file in files if '.npz' in file]
    files = sorted(files, key = lambda x : int(x[x.find('epoch')+5:x.find('epoch')+8]))
    print('Process '+data_path)
    plot_progress = 0
    for file in files:
        print('plot progress:{0}%'.format(round((plot_progress * 100 / len(files)))),end='\r')
        plot_progress+=1
        activation_data_path = os.path.join(data_path,file)
        activation_data = np.load(activation_data_path,allow_pickle=True)
        activations = activation_data['activations']
        pred = activation_data['prediction']
        pred = np.argmax(pred,axis=1)
        data_activation_color = get_data_color(activations,colors)
        act_plot_path = os.path.join(data_path,'activation_plot')
        pred_plot_path = os.path.join(data_path,'prediction_plot')
        if not os.path.exists(act_plot_path):
            os.makedirs(act_plot_path)
        if not os.path.exists(pred_plot_path):
            os.makedirs(pred_plot_path) 
            
        #Plot activation    
        plt.figure(figsize=(10, 10))
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o',s=4, c=data_activation_color)
        plt.title(activation_data_path[:-4]+'activation_'+file[:-4],fontsize=20)
        plt.savefig(act_plot_path+'/'+'activation_'+file[:-4])
        plt.close() 
  
        #Plot prediction
        plt.figure(figsize=(10, 10))
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='o',s=4, c=pred)
        plt.title(activation_data_path[:-4]+'prediction_'+file[:-4],fontsize=20)
        plt.savefig(pred_plot_path+'/'+'prediction_'+file[:-4])
        plt.close() 
