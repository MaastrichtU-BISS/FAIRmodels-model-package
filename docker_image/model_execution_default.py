
from math import log, exp
from model_execution import logistic_regression
import json

class model_execution_logistic_regression(logistic_regression):
    def __init__(self):
        with open('model_parameters.json') as f:
            self._model_parameters = json.load(f)