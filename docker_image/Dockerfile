FROM python:3.11-slim

RUN apt update
RUN apt upgrade -y
RUN apt install -y git

# set a directory for the app
WORKDIR /app

COPY main.py /app/main.py
COPY model_execution_default.py /app/model_execution_default.py

RUN pip install fastapi uvicorn git+https://github.com/MaastrichtU-BISS/FAIRmodels-model-package.git#subdirectory=package

EXPOSE 8000

CMD ["python", "./main.py"]