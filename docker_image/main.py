import os
from typing import List, Union
from fastapi import FastAPI

app = FastAPI()
status_list = []
status_list[0] = "No prediction requested"
status_list[1] = "Prediction requested"
status_list[2] = "Prediction in progress"
status_list[3] = "Prediction completed"
status_list[4] = "Prediction failed"

current_data = None
current_status = 0
current_result = None

def get_model():
    """
    Get the model object based on the environment variables.

    Returns:
    - instance: the model object
    """
    module_name = os.environ.get("MODULE_NAME")
    class_name = os.environ.get("CLASS_NAME")

    # import the module
    module = __import__(module_name)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance

@app.get("/")
def read_root():
    """
    Get the available model metadata
    """
    model_metadata = get_model().get_model_metadata()
    return {
        "model_uri": model_metadata["model_uri"],
        "model_name": model_metadata["model_name"],
        "path": "/predict",	
        "path_parameters": get_model().get_input_parameters(),
    }

@app.post("/predict")
def predict(data: Union[dict, List[dict]]):
    """
    Calculate the probability for the current model.

    Parameters:
    - data: a dictionary (or list of dictionaries) containing the input data
    """
    current_data = data
    status = 1
    model_obj = get_model()
    current_status = 2
    current_result = model_obj.predict(data)
    current_status = 3


@app.get("/status")
def getStatus():
    """
    Get the status of the current model.

    Returns:
    - status: the status of the model
    - message: a message indicating the status
    """
    return {"status": current_status, "message": status_list[current_status]}

@app.get("/result")
def getResult():
    """
    Retrieve the probability for the current model.

    Returns:
    - probability: the probability which the model calculates
    """
    if getStatus()["status"] == 3:
        return current_result
    else:
        return [ ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)