from  aquisition import getData
import loadData 
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

def extract_from_json(json, feature):
    try:
        return json[feature]
    except:
        raise ValueError("Could not find " + feature)

JSON_CONFIGURARE = {
    "DATA_CONFIG": {
            "INPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\NFLX.csv", "COLUMNS": ["close","volume"]},
                    
                ],
            "OUTPUT_FILES": [
                    {"PATH": "C:\\Users\\aalex\\Stocks Project\\historic_data\\NFLX.csv", "COLUMNS": ["close"]},
                ],
            "X_PREVIOUS" : 30,
            "Y_PREDICT" : 3,
        },

    "MODEL_CONFIG": {
            "MODEL": [
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "True"},
                    {"layer_type" : "LSTM", "units" : 40, "activation": "sigmoid", "return_sequences": "False"},
                    {"layer_type" : "DENSE", "units" : 3, "activation": "sigmoid"},
                ],
            "TRAINING_CONFIG": {
                "TENSORBOARD_PATH_CONFIG": "D:\\EXPERIMENTS\\cheie\\NFLX-close-10_3_volume_1",
                "MODEL_SAVED_PATH": "D:\\EXPERIMENTS\\cheie\\model_NFLX-close-10_3_volume_1",
                "EPOCHS" : 20,
                "BATCH_SIZE" : 16,
                "OPTIMIZER" : "ADAM",
                "LEARN_RATE" : 0.001,
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
    training_config = extract_from_json (model_config,"TRAINING_CONFIG")
    model_saved_path=extract_from_json(training_config,"MODEL_SAVED_PATH")

    # Get X and Y dataframe for the model alongside the scaler which are to be saved
    X, Y, X_scalar, Y_scalar = data_provider(data_config)
    
    # Create and train the model
    model=model_configuration(model_config,X.shape,Y.shape)
    train_model(X,Y,model,training_config)

    # Save scalars
    joblib.dump(X_scalar, "X_scalar.pkl")
    joblib.dump(Y_scalar, "Y_scalar.pkl")

    # Save the json
    f = open(model_saved_path+"\\json_model.txt", "w+")
    f.write(json.dumps(json_config))
    f.close()


    X_scaler_new = joblib.load("X_scalar.pkl") 
    Y_scaler_new = joblib.load("Y_scalar.pkl")


    in_files_new = extract_from_json(data_config, "INPUT_FILES")
    x_prev = extract_from_json(data_config, "X_PREVIOUS")



    x_to_predict_on = prediction_preprocessing(in_files_new, x_prev, X_scaler_new) 
    x_to_predict_on = np.reshape(x_to_predict_on, (1, x_to_predict_on.shape[0], x_to_predict_on.shape[1]))
    model_new = tf.keras.models.load_model(model_saved_path)

    predictions = model_new.predict(x_to_predict_on)
    predictions = Y_scaler_new.inverse_transform(predictions)
    print(predictions)













#model_creator(JSON_CONFIGURARE)

if __name__ == "__main__":
    arguments= sys.argv
    print("OK 1")
    # print(sys.argv[1])
    #print(sys.argv[1])
    model_creator(JSON_CONFIGURARE)
    print("GATA")
