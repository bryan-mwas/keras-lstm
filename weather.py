import datetime
from urllib.request import urlopen
import json
import numpy as np

latitude = -1.5177
longitude = 37.2634
start_date = datetime.date.today() - datetime.timedelta(3)
weather_params = 'T2M,WS10M,QV2M,PRECTOTCORR'


def get_weather_json(lookback_days):
    end_date_str = (start_date).strftime("%Y%m%d")
    start_date_str = (
        start_date - datetime.timedelta(lookback_days)).strftime("%Y%m%d")
    url = f'https://power.larc.nasa.gov/api/temporal/daily/point?start={start_date_str}&end={end_date_str}&latitude={latitude}&longitude={longitude}&parameters={weather_params}&community=AG'
    response = urlopen(url)
    data_json = json.loads(response.read())
    return data_json


def get_ndarray_weather():
    data_json = get_weather_json(27)
    weather_param_values = data_json['properties']['parameter']

    response = []

    for param in weather_param_values.keys():
        rows = list(weather_param_values[param].values())
        response.append(rows)

    response_array = np.array(response)
    # Switch rows and columns
    response_array = response_array.transpose()

    return response_array
