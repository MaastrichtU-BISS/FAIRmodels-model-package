import os
import logging
from typing import List, Union
from fastapi import FastAPI, HTTPException

app = FastAPI()
status_list = []
status_list.insert(0, "No prediction requested")
status_list.insert(1, "Prediction requested")
status_list.insert(2, "Prediction in progress")
status_list.insert(3, "Prediction completed")
status_list.insert(4, "Prediction failed")

current_data = None
current_status = 0
current_result = None

logging.basicConfig(level=logging.INFO)


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
    global current_data, current_status, current_result

    current_data = data
    current_status = 1

    try:
        model_obj = get_model()
        current_status = 2
        current_result = model_obj.predict(data)
        current_status = 3
        return current_result

    except (ValueError, TypeError) as e:
        # Validation errors -> client fault (400)
        current_status = 4
        current_result = {"error": str(e)}
        logging.info("Validation error during prediction: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors -> server fault (500)
        current_status = 4
        current_result = {"error": str(e)}
        logging.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/status")
def getStatus():
    """
    Get the status of the current model.

    Returns:
    - status: the status of the model
    - message: a message indicating the status
    """
    if current_status == 4:
        message = current_result.get("error", "") if isinstance(current_result, dict) else ""
        return {"status": current_status, "message": message}
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
        return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)