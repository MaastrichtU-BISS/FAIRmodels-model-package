import os
from typing import List, Union
from fastapi import FastAPI

app = FastAPI()
current_data = None

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

@app.get("/status")
def getStatus():
    """
    Get the status of the current model.

    Returns:
    - status: the status of the model
    - message: a message indicating the status
    """
    return {"status": 0, "message": "No input received yet."}

@app.get("/result")
def getResult():
    """
    Retrieve the probability for the current model.

    Returns:
    - probability: the probability which the model calculates
    """
    if getStatus()["status"] > 0:
        model_obj = get_model()
        return model_obj.predict(current_data)
    else:
        return [ ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)