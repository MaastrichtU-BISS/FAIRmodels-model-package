
import json
from math import exp
import importlib

def load_model(module_name=None, class_name=None):
    """
    Get the model object based on the environment variables.

    Args:
        - module_name (str): The name of the module that contains the model class
        - class_name (str): The name of the class that contains the model
    Returns:
        The model object
    """

    # import the module
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance

def predict(prediction_file: str, class_name: str, input_data: str):
    """
    Predict using a python prediction model execution file, without building the docker image.

    Args:
        prediction_file (str): The python file that contains the prediction model execution code.
            This file should contain a class that inherits from FairModel.model_execution.ModelExecution OR a json file that contains the model parameters
        class_name (str): The name of the class that should inherit from FairModel.model_execution.ModelExecution
        input_data (str): The input data in json formatted string

    Returns:
        The prediction result
    """
    input_data = json.loads(input_data)
    model = None

    if prediction_file.endswith('.json'):
        with open(prediction_file) as f:
            model_parameters = json.load(f)
            if model_parameters['model_type'] == 'logistic_regression':
                model = logistic_regression(model_parameters=model_parameters)
    else:
        module_name = prediction_file.replace('.py', '')
        if class_name is None:
            class_name = module_name

        model = load_model(module_name, class_name)
    
    if model is None:
        print("Model not found")
        return
    
    print(json.dumps(model.predict(input_data), indent=4))

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
            results = { }
            # loop over numeric index of item list
            for i in range(len(input_object)):
                item = input_object[i]
                if "id" in item:
                    results[item["id"]] = self._calculate_probability_single(item)
                else:
                    results[f"prediction_{str(i)}"] = self._calculate_probability_single(item)
            return results

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
            # print(float(input_object[covariate]))
            linear_predictor += float(weight) * float(input_object[covariate])
        
        # Calculate the probability
        probability = 1 / (1 + exp(-(linear_predictor)))
        return probability