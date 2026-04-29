FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# set a directory for the app
WORKDIR /app

COPY main.py /app/main.py
COPY model_execution_default.py /app/model_execution_default.py

RUN pip install --no-cache-dir fastapi uvicorn git+https://github.com/MaastrichtU-BISS/FAIRmodels-model-package.git@error_catch#subdirectory=package

EXPOSE 8000

CMD ["python", "./main.py"]