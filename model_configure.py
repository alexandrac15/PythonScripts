from  aquisition import getData
import pandas as pd 
import tensorflow as tf
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import sys
from data_preprocessing import *
from model_configure import *



def extract_from_json(json, feature):
    try:
        return json[feature]
    except:
        raise ValueError("Could not find " + feature)


def train_model(X,Y,model,training_config):
    tensorboard_path=extract_from_json(training_config,"TENSORBOARD_PATH_CONFIG")
    saved_model_path=extract_from_json(training_config,"MODEL_SAVED_PATH")
    epochs = int(extract_from_json(training_config,"EPOCHS"))
    batch_size = int(extract_from_json(training_config,"BATCH_SIZE"))
    optimizer = extract_from_json(training_config,"OPTIMIZER")
    learn_rate =float( extract_from_json(training_config,"LEARN_RATE"))
    loss = extract_from_json(training_config,"LOSS")
    split=float(extract_from_json(training_config,"SPLIT"))
    
    if optimizer== "SGD":
        optimizer=tf.keras.optimizers.SGD(lr=learn_rate)
    elif optimizer=="ADAM":
        optimizer=tf.keras.optimizers.Adam(lr=learn_rate)

    model.compile(loss = loss,  metrics = ['mse'], optimizer = optimizer)
    x_train,y_train,x_test,y_test=split_data(X,Y,split)

    log_dir=tensorboard_path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    model.fit(x_train, y_train, epochs=epochs,callbacks=[tensorboard_callback],validation_data=(x_test,y_test), verbose = 1, batch_size = batch_size)
  #  model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs,callbacks=[tensorboard_callback],validation_data=(x_test,y_test), verbose = 1)
    return model

def model_configuration(model_config,input_shape,output_shape):
    if len(input_shape) != 3:
        raise ValueError("Invalid input data! It is not 3d")
    if len(output_shape) != 2:
        raise ValueError("Invalid output data! It is not 2d")
    
    model = iterate_json_layers(model_config["MODEL"], input_shape, output_shape)
    print(model.summary())
    return model

def iterate_json_layers(model_blueprint, input_shape, output_shape):
    first_layer = True
    model = tf.keras.Sequential()
    
    for  i in range(0,len(model_blueprint)-1):# layer in model_blueprint:
        layer = model_blueprint[i]
        if first_layer:
            layer_keras = create_layer(layer, input_shape = input_shape)
            first_layer = False
        else:
            layer_keras = create_layer(layer)
        model.add(layer_keras)

    # model.add(tf.keras.layers.Dense(units = output_shape[1]))
    layer = model_blueprint[len(model_blueprint) - 1]
    last_layer = create_layer(layer, output_shape=output_shape)
    model.add(last_layer)
    return model

def create_layer(layer, input_shape = None, output_shape = None):
    print("LAYER: " + json.dumps(layer))
    layer_type = extract_from_json(layer, "layer_type")
    if layer_type=="LSTM":
        units=int(extract_from_json(layer,"units"))
        activation=extract_from_json(layer,"activation")
        return_sequences = (extract_from_json(layer,"return_sequences") == "True")
        if input_shape:
            return tf.keras.layers.LSTM(units=units,activation=activation ,return_sequences=return_sequences, input_shape=(input_shape[1], input_shape[2]))
        else:
            return tf.keras.layers.LSTM(units=units,activation=activation ,return_sequences=return_sequences)
    elif layer_type=="DROPOUT": 
        dropout_rate=int(extract_from_json(layer,"dropout_rate"))
        return tf.keras.layers.Dense(dropout_rate)
    elif layer_type=="DENSE":
        units=int(extract_from_json(layer,"units"))
        activation=extract_from_json(layer,"activation")
        if output_shape:
            return tf.keras.layers.Dense(units = output_shape[1], activation = activation)
        else:
            return tf.keras.layers.Dense(units = units, activation = activation)

       

