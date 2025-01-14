import model_execution
import os
import click
import docker
import json

@click.command()
@click.argument('prediction_file')
@click.argument('image_name')
@click.option('--class_name', default=None, help='The name of the class used in the prediction_python_file')
def build(prediction_file: str, class_name: str, image_name: str):
    """
    Wrap a python prediction model execution file into a container

    Args:
        - prediction_file (str): The python file that contains the prediction model execution code.
            This file should contain a class that inherits from FairModel.model_execution.ModelExecution, OR a json file that contains the model parameters
        - class_name (str): The name of the class that should inherit from FairModel.model_execution.ModelExecution
        - image_name (str): The name of the image that will be created
    """

    # Check if docker is running
    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:
        print("Docker is not running. Please start Docker.")
        return

    if prediction_file.endswith('.json'):
        with open(prediction_file) as f:
            model_parameters = json.load(f)
            module_name = "model_execution_default"
            class_name = f"model_execution_{model_parameters['model_type']}"

            dockerfile = f"""
            FROM jvsoest/base_fairmodels
            WORKDIR /app
            COPY {prediction_file} /app/model_parameters.json
            ENV MODULE_NAME={module_name}
            ENV CLASS_NAME={class_name}
            """
    else:
        module_name = prediction_file.replace('.py', '')
        if class_name is None:
            class_name = module_name
            
            dockerfile = f"""
            FROM jvsoest/base_fairmodels
            WORKDIR /app
            COPY {prediction_file} /app/{prediction_file}
            ENV MODULE_NAME={module_name}
            ENV CLASS_NAME={class_name}
            """

    image = build_container(dockerfile, image_name)

def build_container(dockerfile, image_name, show_logs=False):    
    # write Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    client = docker.from_env()
    # f = BytesIO(dockerfile.encode())
    # print(client.images.build(fileobj=f, rm=True, tag=image_name, path=os.path.abspath(os.path.curdir), custom_context=True))
    image, build_log = client.images.build(path=os.path.abspath(os.path.curdir), rm=True, tag=image_name, nocache=True)
    # delete Dockerfile
    os.remove('Dockerfile')
    if show_logs:
        for line in build_log:
            if 'stream' in line:
                print(line['stream'])
    return image

@click.command()
@click.argument('prediction_python_file')
@click.option('--class_name', default=None, help='The name of the class used in the prediction_python_file')
@click.argument('input_data')
def predict(prediction_python_file: str, class_name: str, input_data: str):
    """
    Predict using a python prediction model execution file, without building the docker image.

    Args:
        prediction_python_file (str): The python file that contains the prediction model execution code.
            This file should contain a class that inherits from FairModel.model_execution.ModelExecution
        class_name (str): The name of the class that should inherit from FairModel.model_execution.ModelExecution
        input_data (str): The input data in json formatted string

    Returns:
        The prediction result
    """
    module_name = prediction_python_file.replace('.py', '')
    if class_name is None:
        class_name = module_name

    model = model_execution.load_model(module_name, class_name)
    return model.predict(json.loads(input_data))