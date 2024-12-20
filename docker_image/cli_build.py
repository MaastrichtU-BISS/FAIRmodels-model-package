from io import BytesIO
import docker
from docker import APIClient
import os

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

#get input arguments
import sys
python_file = sys.argv[1]
image_name = sys.argv[2]
class_name = image_name

# if arguments have 3 elements, the third element is the class name
if len(sys.argv) == 4:
    class_name = sys.argv[3]

module_name = python_file.replace('.py', '')

dockerfile = f"""
FROM jvsoest/base_fairmodels
WORKDIR /app
COPY {python_file} /app/{python_file}
ENV MODULE_NAME={module_name}
ENV CLASS_NAME={class_name}
"""

image = build_container(dockerfile, image_name)