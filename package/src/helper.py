import model_execution
import os
import Union, List

def predict(model_executor: model_execution.ModelExecution, data: Union[dict, List[dict]]):
    """
    Calculate the probability for the current model.

    Parameters:
    - data: a dictionary (or list of dictionaries) containing the input data
    """
    return model_executor.predict(data)