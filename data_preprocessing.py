from  aquisition import getData
import pandas as pd 
import tensorflow as tf
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import sys

def extract_from_json(json, feature):
    try:
        return json[feature]
    except:
        raise ValueError("Could not find " + feature)



def split_data(X,Y,proportion):    
    x_train=X[:(round(len(X)*proportion))]
    y_train=Y[:(round(len(Y)*proportion))]
    #TEST DATA
    x_test=X[(round(len(X)*proportion)):]
    y_test=Y[(round(len(Y)*proportion)):]
    return x_train,y_train,x_test,y_test

def x_normalisation(X):
    scaler=MinMaxScaler(feature_range=(0, 1))
    return  scaler.fit_transform(X), scaler

def normalization(input_dataframe,output_dataframe):
    X_scaled, X_scalar = x_normalisation(input_dataframe)
    Y_scaled, Y_scalar = x_normalisation(output_dataframe)

    return X_scaled,Y_scaled, X_scalar, Y_scalar

def data_provider(data_config):
    input_returned_dataframe, output_returned_dataframe = data_config_function(data_config)
    #print(input_returned_dataframe)
    # print(output_returned_dataframe)
    X_scaled,Y_scaled, X_scalar, Y_scalar = normalization(input_returned_dataframe,output_returned_dataframe)

    X_sequence, Y_sequence = sequencing(X_scaled,Y_scaled,data_config['X_PREVIOUS'], data_config['Y_PREDICT'])

    return X_sequence, Y_sequence, X_scalar, Y_scalar
    # print(X)
    # print(Y)

def sequencing(dataframe,label_column,n_previous,n_next):
    samples=[]
    closing=[]
    vals=label_column

    for i in range(0,len(dataframe)-n_previous-n_next+1):
        samples.append(np.array(dataframe[i:i+n_previous]))
        closing.append(np.array(vals[i+n_previous:i+n_previous+n_next]))

    closing=np.array(closing)
    samples=np.array(samples)  #vector of sliding windows(samples) containing n vectors containing data for 1 day ,example:   [  [[ volume close gdp ...]day1 []day2 ... []day_n ]sample1     []sample2... ]samples
    closing = np.reshape(closing, (len(closing), n_next))

    return samples, closing

def read_data_from_file(file_data):
    try:
        file_path = file_data["PATH"]
    except:
        print("Bad json format: path not found!")
        raise ValueError("File_path not found!")
    dataframe=pd.read_csv(file_path)
    dataframe.drop(dataframe.columns.values[0], axis=1, inplace=True)
    
    try:
        #print(file_data["COLUMNS"])
        returned_dataframe=dataframe[file_data["COLUMNS"]].copy()
    except:
        print("Bad json format: column names not found in csv")
        raise ValueError("column_names not found")

    return returned_dataframe

def iterate_file_list(file_list):
    
    k = 0
    final_dataframe = pd.DataFrame()
    for file_data in file_list:
        #print(file_data)
        obtained_dataframe = read_data_from_file(file_data)
        if final_dataframe.empty:
            final_dataframe = obtained_dataframe
        else:
             #print("joining tables")
             ## TODO: Trim final_dataframe and obtained_dataframe in order to have the same length
             ## Trim the longer dataframe from the beginning
            len_final_dataframe=len(final_dataframe.index)
            len_obtained_dataframe=len(obtained_dataframe.index)
            if len_final_dataframe<len_obtained_dataframe:
                #trim obtained dataframe last: len_final
                obtained_dataframe = obtained_dataframe[len_obtained_dataframe-len_final_dataframe:]
                obtained_dataframe = obtained_dataframe.reset_index(drop = True)
            else:
                final_dataframe = final_dataframe[len_final_dataframe-len_obtained_dataframe:]
                final_dataframe = final_dataframe.reset_index(drop = True)
            final_dataframe = final_dataframe.join(obtained_dataframe, rsuffix = str(k))
        k+=1
    return final_dataframe

def data_config_function(data_config):
    input_file_list = data_config["INPUT_FILES"]
    output_file_list = data_config["OUTPUT_FILES"] 
    ## TODO: Trim input_file_list and output_file_list in order to have the same length
    input_returned_dataframe=iterate_file_list(input_file_list)
    output_returned_dataframe=iterate_file_list(output_file_list)

    len_input_dataframe=len(input_returned_dataframe.index)
    len_output_dataframe=len(output_returned_dataframe.index)
    
    if len_input_dataframe < len_output_dataframe:
        output_returned_dataframe = output_returned_dataframe[len_output_dataframe-len_input_dataframe:]
        output_returned_dataframe = output_returned_dataframe.reset_index(drop = True)
    else:
        input_returned_dataframe = input_returned_dataframe[len_input_dataframe-len_output_dataframe:]
        input_returned_dataframe = input_returned_dataframe.reset_index(drop = True)

    return (input_returned_dataframe, output_returned_dataframe)

def prediction_preprocessing(input_files, x_previous, x_scalar):
    input_returned_dataframe=iterate_file_list(input_files)
    x_scaled = x_scalar.fit_transform(input_returned_dataframe)
    return x_scaled[-x_previous:]



