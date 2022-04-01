from flask import Flask, jsonify
import numpy as np
import pickle
from tensorflow import keras
import weather


scaler = pickle.load(open('scaler.sav', 'rb'))
model = keras.models.load_model('rainfall-model')
app = Flask(__name__)
look_back_steps = 60
no_of_features = 4


# Convert X value to shape required for MinMax scaler i.e (28,4)
def format_model_input(x_input):
    # convert input into numpy array
    data = np.array(x_input)
    data = data.reshape((look_back_steps, no_of_features))
    last_col = np.zeros((look_back_steps, 1))
    return np.append(data, last_col, axis=1)

# Scale the x_input


def scale_transform(df):
    return scaler.fit_transform(df)

# Invert the scaled y_pred


def rescale_y_pred(y_pred):
    y_pred = y_pred.ravel()  # convert from 2D array into 1D array
    # create a new numpy array
    shape = (y_pred.shape[0], no_of_features + 1)
    n_array = np.zeros(shape)
    n_array[:, -1] = y_pred
    return n_array


def reshape_scaled_x_for_prediction(scaled_df):
    X = scaled_df[:, :-1]
    x_shape = (1, look_back_steps, no_of_features)
    return X.reshape(x_shape)


@app.route('/forecast', methods=['GET'])
def rainfall_forecast():
    weather_ndarray = weather.get_ndarray_weather(
        lookback_days=look_back_steps)
    scaled_data = scale_transform(weather_ndarray)
    X_scaled = reshape_scaled_x_for_prediction(scaled_df=scaled_data)
    y_pred = model.predict(X_scaled)
    inv_yhat = scaler.inverse_transform(rescale_y_pred(y_pred))[:, -1]
    rainfall_forecast = inv_yhat.tolist()
    human_readable_forecast = weather.build_human_readable_forecast(
        rainfall_forecast)
    print(len(human_readable_forecast))
    return jsonify(human_readable_forecast)


@app.route('/past-weather', methods=['GET'])
def get_past_weather():
    weather_data = weather.get_ndarray_weather(
        lookback_days=look_back_steps).tolist()
    print(len(weather_data))
    return jsonify(weather_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
