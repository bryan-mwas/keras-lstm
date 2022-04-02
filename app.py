from flask import Flask, jsonify
import weather

app = Flask(__name__)


@app.route('/forecast', methods=['GET'])
def rainfall_forecast():
    human_readable_forecast = weather.forecast_rainfall()
    return jsonify(human_readable_forecast)


@app.route('/past-weather', methods=['GET'])
def get_past_weather():
    weather_data_json = weather.get_weather_json(
        lookback_days=weather.look_back_steps)
    return weather_data_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
