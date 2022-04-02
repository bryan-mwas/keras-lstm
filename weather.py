import json
import pickle
import datetime
import numpy as np
from tensorflow import keras
from urllib.request import urlopen

import weather

scaler = pickle.load(open('scaler.sav', 'rb'))
model = keras.models.load_model('rainfall-model-30-day')
look_back_steps = 60
no_of_features = 4

latitude = -1.5177
longitude = 37.2634
end_date = datetime.date.today() - datetime.timedelta(3)
end_date_str = (end_date).strftime("%Y%m%d")
weather_params = 'T2M,PS,WS10M,QV2M,PRECTOTCORR'


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


def get_weather_json(lookback_days):
    start_date_str = (
        end_date - datetime.timedelta(lookback_days)).strftime("%Y%m%d")
    url = f'https://power.larc.nasa.gov/api/temporal/daily/point?start={start_date_str}&end={end_date_str}&latitude={latitude}&longitude={longitude}&parameters={weather_params}&community=AG'
    response = urlopen(url)
    data_json = json.loads(response.read())
    return data_json


def get_ndarray_weather(lookback_days):
    data_json = get_weather_json(lookback_days - 1)
    weather_param_values = data_json['properties']['parameter']

    response = []

    for param in weather_param_values.keys():
        rows = list(weather_param_values[param].values())
        response.append(rows)

    response_array = np.array(response)
    # Switch rows and columns
    response_array = response_array.transpose()

    return response_array


def build_human_readable_forecast(forecasts):
    forecast_dates = np.array(
        [(end_date + datetime.timedelta(days=i+1)).strftime('%d-%m-%Y') for i in range(len(forecasts))])

    human_readable_forecast = []

    for index, date in enumerate(forecast_dates):
        human_readable_forecast.append({
            "date": date,
            "forecast": forecasts[index]
        })

    return human_readable_forecast


def forecast_rainfall():
    weather_ndarray = get_ndarray_weather(
        lookback_days=look_back_steps)
    scaled_data = scale_transform(weather_ndarray)
    X_scaled = reshape_scaled_x_for_prediction(scaled_df=scaled_data)
    y_pred = model.predict(X_scaled)
    inv_yhat = scaler.inverse_transform(rescale_y_pred(y_pred))[:, -1]
    rainfall_forecast = inv_yhat.tolist()
    human_readable_forecast = weather.build_human_readable_forecast(
        rainfall_forecast)
    print(len(human_readable_forecast))
    return human_readable_forecast
