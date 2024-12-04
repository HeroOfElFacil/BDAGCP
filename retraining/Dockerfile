FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir lightgbm pandas google-cloud-storage

# Add retraining script to the container
COPY retrain.py /app/retrain.py
WORKDIR /app

CMD ["python", "retrain.py"]