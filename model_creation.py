from  aquisition import getData
import loadData 
import pandas as pd 
import tensorflow as tf
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
JSON_CONFIGURARE = {
    
    "DATA_CONFIG": {
            "INPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\NFLX.csv", "COLUMNS": ["close","volume"]},
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\NFLX.csv", "COLUMNS": ["close","volume"]},
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\FB.csv", "COLUMNS": ["close", "volume"]}

                ],
            "OUTPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\NFLX.csv", "COLUMNS": ["close"]},
                ],
            "X_PREVIOUS" : 6,
            "Y_PREDICT" : 3,
        },

    "MODEL_CONFIG": {
            "MODEL": [
                    {"layer_type" : "LSTM", "unts" : 50, "activation": "sigmoid"},
                    {"layer_type" : "LSTM", "unts" : 50, "activation": "sigmoid"},
                    {"layer_type" : "LSTM", "unts" : 50, "activation": "sigmoid"},
                ],
            "EPOCHS" : 1,
            "BATCH_SIZE" : 1,
            "OPTMIZER" : "SGD",
            "LEARN_RATE" : 1,
            "LOSS" : "mse",
        }

}


def model_creator(json):
    data_config = json["DATA_CONFIG"]
    model_config = json["MODEL_CONFIG"]
    input_returned_dataframe, output_returned_dataframe = data_config_function(data_config)
    print(input_returned_dataframe)
    print(output_returned_dataframe)
   


def data_config_function(data_config):
    input_file_list = data_config["INPUT_FILES"]
    output_file_list = data_config["OUTPUT_FILES"] 
    ## TODO: Trim input_file_list and output_file_list in order to have the same length
    input_returned_dataframe=iterate_file_list(input_file_list)
    output_returned_dataframe=iterate_file_list(output_file_list)
    return (input_returned_dataframe, output_returned_dataframe)




def iterate_file_list(file_list):
    
    k = 0
    final_dataframe = pd.DataFrame()
    for file_data in file_list:
        print(file_data)
        obtained_dataframe = read_data_from_file(file_data)
        if final_dataframe.empty:
            final_dataframe = obtained_dataframe
        else:
            print("joining tables")
             ## TODO: Trim final_dataframe and obtained_dataframe in order to have the same length
             ## Trim the longer dataframe from the beginning
            final_dataframe = final_dataframe.join(obtained_dataframe, rsuffix = str(k))
        k+=1
    return final_dataframe


def read_data_from_file(file_data):
    print("read_data_from_file")
    try:
        file_path = file_data["PATH"]
    except:
        print("Bad json format: path not found!")
        raise ValueError("File_path not found!")
    dataframe=pd.read_csv(file_path)
    dataframe.drop(dataframe.columns.values[0], axis=1, inplace=True)
    
    try:
        print(file_data["COLUMNS"])
        returned_dataframe=dataframe[file_data["COLUMNS"]].copy()
    except:
        print("Bad json format: column names not found in csv")
        raise ValueError("column_names not found")

    return returned_dataframe


model_creator(JSON_CONFIGURARE)