from flask import Flask
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

model = keras.models.load_model('rainfall-model')
app = Flask(__name__)

"""Expects a 2D array of 28 rows and 3 columns"""


def format_model_input(x_input):
    # convert input into numpy array
    data = np.array(x_input)
    data = data.reshape((1, data.shape[0], 3))
    return data

# Scale the x_input
# Invert the scaled y_pred
# Convert to shape required for MinMax scaler


@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    a = format_model_input([1, 2, 3, 4])
    return str(a.shape)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
