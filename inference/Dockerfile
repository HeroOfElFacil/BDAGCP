FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
# Install dependencies
RUN pip install --no-cache-dir flask lightgbm google-cloud-pubsub

# Copy the application code
COPY app.py /app/app.py
WORKDIR /app

CMD ["python", "app.py"]
