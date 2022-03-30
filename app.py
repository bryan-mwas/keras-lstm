from flask import Flask, jsonify
import numpy as np
import pickle
from tensorflow import keras


scaler = pickle.load(open('scaler.sav', 'rb'))
model = keras.models.load_model('rainfall-model')
app = Flask(__name__)
look_back_steps = 28
no_of_features = 3

"""Expects a 2D array of 28 rows and 3 columns"""


# Convert X value to shape required for MinMax scaler i.e (28,4)
def format_model_input(x_input):
    # convert input into numpy array
    data = np.array(x_input)
    data = data.reshape((28, 3))
    last_col = np.zeros((28, 1))
    return np.append(data, last_col, axis=1)

# Scale the x_input


def scale_transform(df):
    return scaler.fit_transform(df)

# Invert the scaled y_pred


def rescale_y_pred(y_pred):
    y_pred = y_pred.ravel()  # convert from 2D array into 1D array
    # create a new numpy array
    shape = (y_pred.shape[0], 4)
    n_array = np.zeros(shape)
    n_array[:, -1] = y_pred
    return n_array


def reshape_scaled_x_for_prediction(scaled_df):
    X = scaled_df[:, :-1]
    x_shape = (1, look_back_steps, no_of_features)
    return X.reshape(x_shape)


@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    a = format_model_input(np.random.randint(low=1, high=50, size=28*3))
    scaled_input = scale_transform(a)
    pred = model.predict(
        reshape_scaled_x_for_prediction(scaled_df=scaled_input))
    inv_yhat = rescale_y_pred(pred)[:, -1]
    return jsonify(inv_yhat.tolist())


def get_past_weather():
    pass


def predict_rainfall():
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
