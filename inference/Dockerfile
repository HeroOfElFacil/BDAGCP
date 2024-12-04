FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir flask lightgbm google-cloud-pubsub

# Copy the application code
COPY app.py /app/app.py
WORKDIR /app

CMD ["python", "app.py"]
