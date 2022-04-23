from flask import Flask, jsonify, request
from flask_cors import CORS
import weather

app = Flask(__name__)
CORS(app)

@app.route('/forecast', methods=['GET'])
def rainfall_forecast():
    latitude = request.args.get('lat')
    longitude = request.args.get('lng')
    human_readable_forecast = weather.forecast_rainfall(lat=latitude, lng=longitude)
    return jsonify(human_readable_forecast)


@app.route('/past-weather', methods=['GET'])
def get_past_weather():
    print(request.args.get('lat'))
    latitude = request.args.get('lat')
    longitude = request.args.get('lng')
    weather_data_json = weather.get_weather_json(
        lookback_days=weather.look_back_steps, latitude=latitude, longitude=longitude)
    return weather_data_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
