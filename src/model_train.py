"""
SageMaker-compatible training script for Customer Sentiment Analysis
"""

import os
import argparse
import json
import pandas as pd
import numpy as np

# --------------------------------------------------
# Argument parsing (UPDATED to accept all workflow arguments)
# --------------------------------------------------
parser = argparse.ArgumentParser()
# Accept the hyperparameters passed from the GitHub Actions workflow
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--bucket", type=str)
parser.add_argument("--project", type=str)
parser.add_argument("--region", type=str)
args = parser.parse_args()

# --------------------------------------------------
# Mapping arguments to variables
# --------------------------------------------------
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
S3_BUCKET = args.bucket
PROJECT_NAME = args.project
AWS_REGION = args.region

print("========== TRAINING CONFIG ==========")
print(f"S3_BUCKET      : {S3_BUCKET}")
print(f"PROJECT_NAME   : {PROJECT_NAME}")
print(f"AWS_REGION     : {AWS_REGION}")
print(f"EPOCHS         : {EPOCHS}")
print(f"BATCH_SIZE     : {BATCH_SIZE}")
print("=====================================")

# --------------------------------------------------
# SageMaker paths
# --------------------------------------------------
# Input data (mounted automatically by SageMaker)
TRAINING_DATA_DIR = "/opt/ml/input/data/training"

# Output model directory (SageMaker expects artifacts here)
MODEL_DIR = "/opt/ml/model"

# --------------------------------------------------
# Load training data
# --------------------------------------------------
print("Loading training data...")

data_files = os.listdir(TRAINING_DATA_DIR)
if not data_files:
    raise RuntimeError("No training data found in SageMaker input channel")

data_path = os.path.join(TRAINING_DATA_DIR, data_files[0])
df = pd.read_csv(data_path)

print(f"Loaded dataset shape: {df.shape}")

# --------------------------------------------------
# Dummy preprocessing (replace with real logic)
# --------------------------------------------------
print("Running preprocessing...")

X = df.drop(columns=["label"], errors="ignore")
y = df["label"] if "label" in df.columns else np.zeros(len(df))

# --------------------------------------------------
# Dummy training loop (replace with real model)
# --------------------------------------------------
print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS} - training step complete")

print("Training finished.")

# --------------------------------------------------
# Save model artifacts (REQUIRED by SageMaker)
# --------------------------------------------------
print("Saving model artifacts...")

model_artifact = {
    "model_type": "dummy-sentiment-model",
    "epochs": EPOCHS,
    "features": list(X.columns),
}

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "model.json"), "w") as f:
    json.dump(model_artifact, f)

print("Model artifacts saved successfully.")
print("Training job completed without errors.")
