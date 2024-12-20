FROM python:3.11-slim

# set a directory for the app
WORKDIR /app

COPY main.py /app/main.py
COPY model_execution.py /app/model_execution.py

RUN pip install fastapi uvicorn

EXPOSE 8000

CMD ["python", "./main.py"]