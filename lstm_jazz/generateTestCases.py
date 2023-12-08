from .solutions import *
import numpy as np

np.random.seed(3)

import math

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Reshape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_a = 64
n_values = 90
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')
x_initializer = np.zeros((1, 1, 90))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
reshapor = Reshape((1, n_values))

inference_model = music_inference_model(LSTM_cell, densor, 13)
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)


def generateTestCases():
    testCases = {
        'djmodel': {
            'partId': 'iz6sX',
            'testCases': [
                {
                    'testInput': (30, LSTM_cell, densor, reshapor),
                    'testOutput': (45530, 36)
                }
            ]
        },
        'music_inference_model': {
            'partId': 'MtuL2',
            'testCases': [
                {
                    'testInput': (LSTM_cell, densor, 10),
                    'testOutput': (45530, 32)
                }
            ]
        },
        'predict_and_sample': {
            'partId': 'tkaiA',
            'testCases': [
                {
                    'testInput': (inference_model, x_initializer, a_initializer, c_initializer),
                    'testOutput': (results, indices)
                }
            ]
        },
    }
    return testCases
