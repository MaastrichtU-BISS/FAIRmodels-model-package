
import json
from math import exp

class model_execution:
    def get_model_metadata(self):
        return {
            "model_name": None,
            "model_uri": None,
        }

    def get_input_parameters(self):
        """
        Get the input parameters of the model.

        Returns:
        - input_parameters: a list of input parameters
        """

        return None

    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """

        return data

    def _calculate_probability_single(self, data):
        """
        Calculate the probability of 2-year survival for a patient with given covariates.

        Parameters:
        - input_object: a dictionary containing the input data
        """
        
        return None
    
    def predict(self, input_object):
        """
        Calculate the probability of 2-year survival for a patient with given covariates.

        Parameters:
        - input_object: a dictionary or list containing the input data
        """
        
        # Preprocess the input data
        input_object = self._preprocess(input_object)

        # Calculate the probability
        if isinstance(input_object, dict):
            return self._calculate_probability_single(input_object)
        elif isinstance(input_object, list):
            return [self._calculate_probability_single(item) for item in input_object]

class logistic_regression(model_execution):
    def __init__(self, model_parameters=None, model_path=None):
        self._model_parameters = None
        if model_path is not None:
            self._model_parameters = json.load(open(model_path, 'r'))
        if model_parameters is not None:
            self._model_parameters = model_parameters
        
        if self._model_parameters is None:
            raise ValueError("Model parameters not provided")
    
    def get_model_metadata(self):
        return {
            "model_name": self._model_parameters['model_name'],
            "model_uri": self._model_parameters['model_uri'],
        }

    def get_input_parameters(self):
        """
        Get the input parameters of the model.

        Returns:
        - input_parameters: a list of input parameters
        """
        return list(self._model_parameters['covariate_weights'].keys())
    
    def _calculate_probability_single(self, input_object):
        """
        Calculate probability for logistic regression.

        Parameters:
        - input_object: a dictionary containing the input data
        """
        
        # Calculate the linear predictor
        linear_predictor = self._model_parameters['intercept']
        for covariate, weight in self._model_parameters['covariate_weights'].items():
            print(float(input_object[covariate]))
            linear_predictor += float(weight) * float(input_object[covariate])
        
        # Calculate the probability
        probability = 1 / (1 + exp(-(linear_predictor)))
        return probability