import os
import logging
from typing import List, Union
from fastapi import FastAPI, HTTPException, Request
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


# Global exception handlers so validation or other Value/Type errors
# raised anywhere in the app will set the global status and return 400.
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    global current_status, current_result
    logging.warning("Request validation error: %s", exc)
    current_status = 4
    current_result = {"error": str(exc)}
    # Return 400 to indicate client error (you can keep 422 if you prefer)
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

        # model_obj.predict may raise ValueError/TypeError/KeyError which we want to convert
        # to 400 responses and update global status/result.
        try:
            current_result = model_obj.predict(data)
        except (ValueError, TypeError, KeyError) as e:
            # Validation errors -> client fault (400)
            logging.info("Validation error in model.predict: %s", e)
            current_status = 4
            current_result = {"error": str(e)}
            # raise HTTPException so FastAPI returns a proper HTTP response and our handlers run
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Unexpected server error -> 500
            logging.exception("Unhandled exception in model.predict")
            current_status = 4
            current_result = {"error": str(e)}
            raise HTTPException(status_code=500, detail="Internal server error")

        # success
        current_status = 3
        return current_result

    except HTTPException:
        # re-raise HTTP exceptions (they were already handled and status/result set)
        raise
    except Exception as e:
        # Anything else that slipped through
        logging.exception("Unhandled exception in predict endpoint")
        current_status = 4
        current_result = {"error": str(e)}
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
        return {"status": current_status, "message": current_result.get("error", "")}
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