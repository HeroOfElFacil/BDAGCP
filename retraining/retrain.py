import pandas as pd
import lightgbm as lgb
from google.cloud import storage

# Cloud Storage bucket details
BUCKET_NAME = "your-bucket-name"
TRAINING_DATA = "data/training.csv"
MODEL_OUTPUT = "models/latest_model.txt"

def download_training_data():
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(TRAINING_DATA)
    blob.download_to_filename("/tmp/training.csv")

def upload_model(local_path):
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_OUTPUT)
    blob.upload_from_filename(local_path)

def retrain():
    # Download training data
    #download_training_data()
    print("dupa")
    # Load training data
    data = pd.read_parquet("../data/training_data.parquet")
    X = data.drop("totalvolume", axis=1)
    y = data["totalvolume"]

    # Train model
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "regression", "metric": "binary_error"}
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Save the model locally and upload to Cloud Storage
    model.save_model("../models/latest_model.txt")
    #upload_model("/tmp/latest_model.txt")

if __name__ == "__main__":
    retrain()
