from flask import Flask, request, jsonify
import lightgbm as lgb
import base64
import json
import logging
import fastavro
from google.cloud import storage
import os
from io import BytesIO

app = Flask(__name__)

# List of feature columns
FEATURE_COLUMNS = ['trp_lat', 'trp_lon', 'air_temperature', 'precipitation_rate',
       'relative_humidity', 'wind_from_direction', 'wind_speed',
       'wind_speed_of_gust', 'totalvolume', 'sin_measurement_time',
       'cos_measurement_time']

# Define the Avro schema
AVRO_SCHEMA = {
    "type": "record",
    "name": "Prediction",
    "fields": [{"name": col, "type": "float"} for col in FEATURE_COLUMNS] + 
              [{"name": "prediction", "type": "float"}]
}

# Function to save data to GCS as Avro
def save_to_avro(bucket_name, blob_name, data, schema):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Use BytesIO for writing Avro data
    with BytesIO() as buffer:
        fastavro.writer(buffer, schema, data)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

# Google Cloud Storage bucket and model path
BUCKET_NAME = "seraphic-result"
MODEL_FILE = "models/latest_model.txt"

def download_model():
    """Download the model from GCS to a local file."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    local_model_path = "/tmp/latest_model.txt"
    blob.download_to_filename(local_model_path)
    return local_model_path

# Load the model during startup
MODEL_PATH = download_model()
model = lgb.Booster(model_file=MODEL_PATH)
# TODO: take into account properly converting the categorical data
@app.route("/predict", methods=["POST"])
def predict():
    # Get the Pub/Sub message
    envelope = request.get_json()
    
    # Check if the message is valid
    if not envelope or 'message' not in envelope:
        logging.log("invalid message format")
        return jsonify({'error': 'Invalid message format'}), 400

    # Decode the message data
    data = envelope['message']['data']
    decoded_data = base64.b64decode(data).decode('utf-8')

    # Process the decoded message
    # Assuming it's JSON
    features = json.loads(decoded_data)['features']
    if 'features' not in features or len(features['features']) != len(FEATURE_COLUMNS):
        return jsonify({'error': 'Feature length mismatch'}), 400
    
    # Run prediction
    prediction = model.predict([features])
    
    # Combine features with the prediction
    result = dict(zip(FEATURE_COLUMNS, features['features']))
    result['prediction'] = prediction

    # Save result to an Avro file
    blob_name = 'predictions/prediction.avro'

    # Convert to a list (Avro expects a list of records)
    save_to_avro(BUCKET_NAME, blob_name, [result], AVRO_SCHEMA)

    # Return the response
    logging.log(f"Prediction saved to GCS; prediction: {prediction}")
    return jsonify({'message': 'Prediction saved to GCS', 'prediction': prediction}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
