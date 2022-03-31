import datetime
from urllib.request import urlopen
import json
import numpy as np

latitude = -1.5177
longitude = 37.2634
end_date = datetime.date.today() - datetime.timedelta(3)
end_date_str = (end_date).strftime("%Y%m%d")
weather_params = 'T2M,WS10M,QV2M,PRECTOTCORR'


def get_weather_json(lookback_days):
    start_date_str = (
        end_date - datetime.timedelta(lookback_days)).strftime("%Y%m%d")
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


def build_human_readable_forecast(forecasts):
    forecast_dates = np.array(
        [(end_date + datetime.timedelta(days=i)).strftime('%d-%m-%Y') for i in range(len(forecasts))])

    human_readable_forecast = []

    for index, date in enumerate(forecast_dates):
        human_readable_forecast.append({
            "date": date,
            "forecast": forecasts[index]
        })

    return human_readable_forecast
