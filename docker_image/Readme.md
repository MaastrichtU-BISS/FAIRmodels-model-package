**This folder needs a separate repository and a PyPi package, as this needs to be made public as part of FAIRmodels**

# Base image

This folder contains the base image. The files needed in this base image are:

- [main.py](main.py)
- [model_execution.py](model_execution.py)

The [model_execution.py](model_execution.py) file also acts as a module containing the base class for the model classes.

# Building an image

The [cli_build.py](cli_build.py) file holds a script to build the docker image, based on the base image.
This script can be called as follows:

```
python cli_build.py <model_filename.py> <docker_image_name> <optional_class_name>
```

The final `optional_class_name` is only needed when the module name (read: the model_filename) is not equal to the class name within the file.