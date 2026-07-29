import model_execution
import os
import click
import docker
import json
from pathlib import Path


@click.command()
@click.argument('prediction_file')
@click.argument('image_name')
@click.option('--class_name', default=None, help='The name of the class used in the prediction_python_file')
@click.option('--requirements', type=click.Path(exists=True), default=None,
              help='Path to requirements.txt file with custom dependencies')
@click.option('--dockerfile', type=click.Path(exists=True), default=None,
              help='Optional custom Dockerfile path. If omitted, fm-build generates a default Dockerfile.')
@click.option('--context', type=click.Path(exists=True), default=None,
              help='Docker build context path (default: current directory).')
def build(prediction_file: str, image_name: str, class_name: str = None, requirements: str = None, 
          dockerfile: str = None, context: str = None):
    """
    Wrap a python prediction model execution file into a container
    """

    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:
        print("Docker is not running. Please start Docker.")
        return

    # If custom dockerfile is provided, use it directly
    if dockerfile:
        build_with_custom_dockerfile(dockerfile, image_name, context or os.path.abspath(os.path.curdir), show_logs=True)
        return

    # Otherwise, generate default dockerfile
    if prediction_file.endswith('.json'):
        # ... existing JSON logic ...
        dockerfile_content = f"""
        FROM ghcr.io/maastrichtu-biss/fairmodels-model-package/base-image:latest
        WORKDIR /app
        COPY {prediction_file} /app/model_parameters.json
        ENV MODULE_NAME={module_name}
        ENV CLASS_NAME={class_name}
        """
    else:
        module_name = prediction_file.replace('.py', '')
        if class_name is None:
            class_name = module_name

        # Handle requirements file
        requirements_cmd = ""
        if requirements:
            requirements_cmd = f"COPY {requirements} /app/requirements.txt\nRUN pip install -r /app/requirements.txt\n"

        dockerfile_content = f"""
        FROM ghcr.io/maastrichtu-biss/fairmodels-model-package/base-image:latest
        WORKDIR /app
        {requirements_cmd}
        COPY {prediction_file} /app/{prediction_file}
        ENV MODULE_NAME={module_name}
        ENV CLASS_NAME={class_name}
        """

    image = build_container(dockerfile_content, image_name, show_logs=True)

def build_container(dockerfile, image_name, show_logs=False):    
    # write Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    client = docker.from_env()

    # build image
    image, build_log = client.images.build(path=os.path.abspath(os.path.curdir), rm=True, tag=image_name, nocache=True)
    
    # delete Dockerfile
    os.remove('Dockerfile')
    if show_logs:
        for line in build_log:
            if 'stream' in line:
                print(line['stream'])
    return image

def build_with_custom_dockerfile(dockerfile_path: str, image_name: str, context: str, show_logs=False):
    """
    Build docker image using a custom Dockerfile.
    
    Args:
        dockerfile_path: Path to the custom Dockerfile
        image_name: Docker image tag
        context: Docker build context path
        show_logs: Whether to show build logs
    """
    client = docker.from_env()
    context_abs = os.path.abspath(context)
    dockerfile_abs = os.path.abspath(dockerfile_path)
    
    # Validate paths
    if not os.path.exists(context_abs):
        raise FileNotFoundError(f"Build context does not exist: {context_abs}")
    if not os.path.exists(dockerfile_abs):
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_abs}")
    
    # Get relative path from context to dockerfile
    dockerfile_rel = os.path.relpath(dockerfile_abs, context_abs)
    
    # build image using custom dockerfile
    try:
        image, build_log = client.images.build(
            path=context_abs,
            dockerfile=dockerfile_rel,
            rm=True,
            tag=image_name,
            nocache=True
        )
        
        if show_logs:
            for line in build_log:
                if 'stream' in line:
                    print(line['stream'])
                if 'error' in line:
                    print(f"ERROR: {line['error']}")
        return image
    except docker.errors.BuildError as e:
        print(f"\n=== Docker Build Failed ===")
        print(f"Error: {str(e)}")
        if hasattr(e, 'build_log'):
            print("\n=== Build Log ===")
            for line in e.build_log:
                if 'stream' in line:
                    print(line['stream'], end='')
                if 'error' in line:
                    print(f"ERROR: {line['error']}")
        raise

@click.command()
@click.argument('prediction_file')
@click.option('--class_name', default=None, help='The name of the class used in the prediction_python_file')
@click.argument('input_data')
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
                model = model_execution.logistic_regression(model_parameters=model_parameters)
    else:
        module_name = prediction_file.replace('.py', '')
        if class_name is None:
            class_name = module_name

        model = model_execution.load_model(module_name, class_name)
    
    if model is None:
        print("Model not found")
        return
    
    print(json.dumps(model.predict(input_data), indent=4))
