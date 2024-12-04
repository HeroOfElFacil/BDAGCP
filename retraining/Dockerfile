FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
# Install dependencies
RUN pip install --no-cache-dir lightgbm pandas google-cloud-storage

# Add retraining script to the container
COPY retrain.py /app/retrain.py
WORKDIR /app

CMD ["python", "retrain.py"]