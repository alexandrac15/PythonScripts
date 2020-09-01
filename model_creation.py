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
from sklearn.externals import joblib 
import os

def extract_from_json(json, feature):
    try:
        return json[feature]
    except:
        raise ValueError("Could not find " + feature)

JSON_CONFIGURARE = {
    "DATA_CONFIG": {
            "INPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks_Project\\historic_data\\MEET.csv", "COLUMNS": ["close", "volume"]},
                    # {"PATH": "C:\\Users\\aalex\\Stocks_Project\\historic_data\\FB.csv", "COLUMNS": ["close", "volume"]},
                    
                ],
            "OUTPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks_Project\\historic_data\\MEET.csv", "COLUMNS": ["close"]},
                ],
            "X_PREVIOUS" : 120,
            "Y_PREDICT" : 3,
        },

    "MODEL_CONFIG": {
            "MODEL": [
                    # {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    # {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                   
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "False"},
                    {"layer_type" : "DENSE", "units" : 3, "activation": "sigmoid"},
                ],
            "TRAINING_CONFIG": { 
                "TENSORBOARD_PATH_CONFIG": "D:\\EXPERIMENTS\\E_INDUSTRY\\INTERNET_SOFTWARE\\MEET_CLOSE_TB_3_120",
                "MODEL_SAVED_PATH": "D:\\EXPERIMENTS\\E_INDUSTRY\\INTERNET_SOFTWARE\\MEET_CLOSE_3_120",
                "EPOCHS" : 50,
                "BATCH_SIZE" : 12,
                "OPTIMIZER" : "NADAM",
                "LEARN_RATE" : 0.00115,
                "LOSS" : "mse",
                "SPLIT": 0.9
            }
        }   
}


def model_creator(json_config):
    print(" ----- model_creator ------")

    # Extract data
    data_config = json_config["DATA_CONFIG"]
    model_config = json_config["MODEL_CONFIG"]

    print("OUT: " + str(extract_from_json(data_config, "Y_PREDICT")))
    print("IN: " + str(extract_from_json(data_config, "X_PREVIOUS")))


    training_config = extract_from_json (model_config,"TRAINING_CONFIG")
    print("BATCh SIZE: " + str(extract_from_json(training_config, "BATCH_SIZE")))

    model_saved_path=extract_from_json(training_config,"MODEL_SAVED_PATH")

    # Get X and Y dataframe for the model alongside the scaler which are to be saved
    X, Y, X_scalar, Y_scalar = data_provider(data_config)
    
    print("###########################################")
    print("shape x = " + json.dumps(X.shape))
    print("shape y = " + json.dumps(Y.shape))
    print("###########################################")

    # Create and train the model
    model=model_configuration(model_config,X.shape,Y.shape)
    trained_model = train_model(X,Y,model,training_config)

    # Save model
    trained_model.save(model_saved_path) 

    # print(model_saved_path)
    # if not os.path.exists(model_saved_path+"\\"):
    #     os.mkdir(model_saved_path+"\\", mode=0x777)


    # Save scalars
    joblib.dump(X_scalar, model_saved_path + "X_scalar.pkl")
    joblib.dump(Y_scalar, model_saved_path + "Y_scalar.pkl")

    # Save the json
    try:
        f = open(model_saved_path+"\json_model.txt", 'w+')
        f.write(json.dumps(json_config))
        f.close()
    except:
        pass

    X_scaler_new = joblib.load(model_saved_path + "X_scalar.pkl") 
    Y_scaler_new = joblib.load(model_saved_path + "Y_scalar.pkl")


    in_files_new = extract_from_json(data_config, "INPUT_FILES")
    x_prev = extract_from_json(data_config, "X_PREVIOUS")



    x_to_predict_on = prediction_preprocessing(in_files_new, x_prev, X_scaler_new) 
    x_to_predict_on = np.reshape(x_to_predict_on, (1, x_to_predict_on.shape[0], x_to_predict_on.shape[1]))
    model_new = tf.keras.models.load_model(model_saved_path)

    predictions = model_new.predict(x_to_predict_on)
    predictions = Y_scaler_new.inverse_transform(predictions)
    print(predictions)



if __name__ == "__main__":
    if len(sys.argv) >= 2: 
        arguments= sys.argv
        data = json.loads(sys.argv[1])
        model_creator(data)
    else:
        model_creator(JSON_CONFIGURARE)
    print("GATA")

