import os
import logging
from typing import List, Union, Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()
logging.basicConfig(level=logging.INFO)

status_list = []
status_list.insert(0, "No prediction requested")
status_list.insert(1, "Prediction requested")
status_list.insert(2, "Prediction in progress")
status_list.insert(3, "Prediction completed")
status_list.insert(4, "Prediction failed")

current_data: Any = None
current_status = 0
current_result: Any = None


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


# Keep handlers for validation errors that occur before the endpoint runs.
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    global current_status, current_result
    logging.warning("Request validation error: %s", exc)
    current_status = 4
    current_result = {"error": str(exc)}
    # still return a 400 for the initial request; /status and /result reflect the error state
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    global current_status, current_result
    logging.warning("ValueError: %s", exc)
    current_status = 4
    current_result = {"error": str(exc)}
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    global current_status, current_result
    logging.warning("TypeError: %s", exc)
    current_status = 4
    current_result = {"error": str(exc)}
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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
    Accept a prediction request, update global status/result, and return {}.
    The caller must poll /status and /result to see outcome.
    """
    global current_data, current_status, current_result

    current_data = data
    current_status = 1  # request received

    try:
        model_obj = get_model()
    except Exception as e:
        logging.exception("Failed to instantiate model")
        current_status = 4
        current_result = {"error": str(e)}
        # Do not raise; return empty body. Status/result reflect the failure.
        return {}

    current_status = 2  # prediction in progress

    try:
        # Run prediction synchronously and store result or error in globals.
        result = model_obj.predict(data)
        current_result = result
        current_status = 3  # completed
        # Do not return the result here — clients are expected to fetch /result
        return {}

    except (ValueError, TypeError, KeyError) as e:
        # Validation/client errors -> set failed state and store error message
        logging.info("Validation error in model.predict: %s", e)
        current_status = 4
        current_result = {"error": str(e)}
        return {}

    except Exception as e:
        # Unexpected server error -> set failed state and store error (traceback in logs)
        logging.exception("Unhandled exception in model.predict")
        current_status = 4
        current_result = {"error": str(e)}
        return {}


@app.get("/status")
def getStatus():
    """
    Get the status of the current model run.
    Returns:
    - status: numeric status code
    - message: human-readable message or error
    """
    if current_status == 4:
        return {"status": current_status, "message": current_result.get("error", "")}
    return {"status": current_status, "message": status_list[current_status]}


@app.get("/result")
def getResult():
    """
    Retrieve the model output if ready.
    """
    if getStatus()["status"] == 3:
        return current_result
    else:
        return {}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)