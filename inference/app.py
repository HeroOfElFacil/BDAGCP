from flask import Flask, request, jsonify
import lightgbm as lgb
from google.cloud import storage
import os

app = Flask(__name__)

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
    try:
        # Parse JSON input
        data = request.get_json()
        features = data["features"]
        
        # Run prediction
        prediction = model.predict([features])
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
