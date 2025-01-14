# FAIRmodels-model-package

This package is intended as a base for all models available on FAIRmodels.org.
As a model developer, models need to be encapsulated into Docker containers. The instructions below will help with convenience scripts.
To build your own container from scratch, a REST API is required according to the specification given in [docker_image/api-specification.yaml](docker_image/api-specification.yaml). However, for 80% of the models, it is easier to perform the steps below.

## 1. Install the package

To install the software, you first need to install a Python package, given the command below.
```
pip install git+https://github.com/MaastrichtU-BISS/FAIRmodels-model-package.git#subdirectory=package
```

## 2. Write the model specification

There are two ways you can write a model specification:
1. Using a JSON specification
2. Writing a Python function/script

Both options are explained below.

### 2.1 Using a JSON specification

Currently the JSON specification only works for Logistic Regression models, where no transformations are necessary. For example, we will have the following model specification
```json
{
    "model_type": "logistic_regression",
    "model_uri": "https://v2.fairmodels.org/instance/3f400afb-df5e-4798-ad50-0687dd439d9b",
    "model_name": "Stiphout pCR prediction - clinical parameters",
    "intercept": -0.6,
    "covariate_weights": {
        "cT": -0.074,
        "cN": -0.060,
        "tLength": -0.085
    }
}
```

The following information properties are defined in this JSON specification:
- model_type: the type of the prediction model (currently, only Logistic Regression without transformations are supported)
- model_uri: The unique identifier of the prediction model. E.g. a reference to a FAIRmodel description.
- model_name: The human-readable name of the prediction model
- intercept: the intercept value for the logistic regression
- covariate_weights: a dictionary containing the covariate weights, where the key is the variable name, and the value is the actual weight.

### 2.2 Using a python function/script

If the above option is too limited, you can use the python function/script option to make your own (custom) model. This model class should inherit from [package/src/model_execution.py](model_execution.py). The same prediction model is shown below as a python function/script:

```python
from math import log, exp
from model_execution import logistic_regression

class stiphout_pCR_clinical(logistic_regression):
    def __init__(self):
        self._model_parameters = {
            "model_uri": "https://v2.fairmodels.org/instance/3f400afb-df5e-4798-ad50-0687dd439d9b",
            "model_name": "Stiphout pCR prediction - clinical parameters",
            "intercept": -0.6,
            "covariate_weights": {
                "cT": -0.074,
                "cN": -0.060,
                "tLength": -0.085
            }
        }
    
    def _preprocess(self, data):
        """
        This function is used to convert the input data into the correct format for the model.

        Parameters:
        - input_object: a dictionary, or list with multiple dictionaries, containing the input data

        Returns:
        - preprocessed_data: a dictionary, or list with multiple dictionaries, containing the preprocessed data
        """

        # perform log transformation on the gtv value which is in the data list/dictionary
        if isinstance(data, list):
            for i in range(len(data)):
                data[i]['tLength'] = log(data[i]['tLength'])
        else:
            data['tLength'] = log(data['tLength'])
        
        return data
```

In this example, the variable `tLength` is transformed into the value `log(tLength)`.

Another example could highlight a custom prediction function, in this case, the following code could apply.

```python
class deep_thought(model_execution):
    def __init__(self):
        self._model_parameters = {
            "model_uri": "https://www.wikidata.org/wiki/Q3107329",
            "model_name": "Prediction of the meaning of life, the universe and everything"
        }
    
    def predict(self, data):
        """
        Calculate the probability for the current model.

        Parameters:
        - data: a dictionary (or list of dictionaries) containing the input data
        """
        return 42
    
    def get_input_parameters(self):
        """
        Get the input parameters of the model.
        """
        return ["question"]
```

## 3. Build the container

To build the container, you can use the command-line convenience scripts. This command-line script works with both the JSON specification and python function/script.
The examples below are based on the specifications/scripts above.

To build/package using the JSON specification:
```
fm-build stiphout_pcr_clinical.json jvsoest/stiphout_pcr_clinical
```

In this example, the first argument (`stiphout_pcr_clinical.json`) refers to the JSON file. The second argument (`jvsoest/stiphout_pcr_clinical`) refers to the Docker image name which will be built.

To build/package using the python function/script:
```
fm-build deep_thought.py --class_name deep_thought jvsoest/deep_thought
```

In this example, the syntax is similar, with the additional `--class_name` argument, which specifies the class definition in the python script file.

## 4. Execute the container and run predictions

After the image is built, the predictions can be executed. First we need to start the container, whereafter we can provide a HTTP-REST call to the model for inferencing.

### 4.1 start the container

Below is an example how to execute the container

```
docker run --rm -p 8000:8000 jvsoest/stiphout_pcr_clinical
```

When opening [http://localhost:8000](http://localhost:8000), the local model metadata will be shown in JSON format.

### 4.2 Perform prediction

The prediction can be called using the `/predict` path in the URL. The given CURL example will show how this can be called
```bash
curl -X POST http://localhost:8000/predict -d '{\"cT\": 3, \"cN\": 1, \"tLength\": 7}'
```

Afterwards, the status of the prediction can be fetched using the following CURL request
```bash
curl http://localhost:8000/status
```

Finally, if the status identifies there is a prediction result (status id=3), the result can be fetched
```bash
curl http://localhost:8000/result
```