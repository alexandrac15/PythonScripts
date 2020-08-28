from sklearn.externals import joblib
import tensorflow as tf
import numpy as np 
from data_preprocessing import *

def extract_from_json(json, feature):
    try:
        return json[feature]
    except:
        raise ValueError("Could not find " + feature)


def read_model_file(model_path):
    try:
        json_file = open(model_path + "\\json_model.txt", "r")
    except Exception as e:
        print("Could not open json model config file")
        raise e
    json_config = json_file.read()
    json_file.close()
    predictions = make_prediction(json.loads(json_config))
    print(predictions)


def make_prediction(json_config):
    data_config = json_config["DATA_CONFIG"]
    model_config = json_config["MODEL_CONFIG"]
    training_config = extract_from_json (model_config,"TRAINING_CONFIG")
    model_saved_path=extract_from_json(training_config,"MODEL_SAVED_PATH")

    X_scaler_new = joblib.load(model_saved_path + "X_scalar.pkl")   
    Y_scaler_new = joblib.load(model_saved_path + "Y_scalar.pkl")


    in_files_new = extract_from_json(data_config, "INPUT_FILES")
    x_prev = extract_from_json(data_config, "X_PREVIOUS")



    x_to_predict_on = prediction_preprocessing(in_files_new, x_prev, X_scaler_new)

    x_to_predict_on = np.reshape(x_to_predict_on, (1, x_to_predict_on.shape[0], x_to_predict_on.shape[1]))
    model_new = tf.keras.models.load_model(model_saved_path)


    predictions = model_new.predict(x_to_predict_on)
    predictions = Y_scaler_new.inverse_transform(predictions)
    return predictions


if __name__ == "__main__":
    arguments= sys.argv
    model_path = sys.argv[1]
    read_model_file(model_path)


