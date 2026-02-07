"""
SageMaker-compatible training script for Customer Sentiment Analysis
(Proper HuggingFace model saving)
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--bucket", type=str)
parser.add_argument("--project", type=str)
parser.add_argument("--region", type=str)

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

print("========== TRAINING CONFIG ==========")
print(f"EPOCHS      : {EPOCHS}")
print(f"BATCH_SIZE : {BATCH_SIZE}")
print("=====================================")

# -------------------------------
# SageMaker paths
# -------------------------------
TRAINING_DATA_DIR = "/opt/ml/input/data/training"
MODEL_DIR = "/opt/ml/model"

# -------------------------------
# Load training data
# -------------------------------
print("📥 Loading training data...")

files = os.listdir(TRAINING_DATA_DIR)
if not files:
    raise RuntimeError("No training data found")

data_path = os.path.join(TRAINING_DATA_DIR, files[0])
df = pd.read_csv(data_path)

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Training data must contain 'text' and 'label' columns")

print(f"✅ Dataset shape: {df.shape}")

# -------------------------------
# Load tokenizer & model
# -------------------------------
print("🤖 Loading DistilBERT...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(df["label"].unique())
)

# -------------------------------
# Dataset wrapper
# -------------------------------
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TweetDataset(
    df["text"].tolist(),
    df["label"].tolist()
)

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="/tmp/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=500,                 # ⬅️ HARD STOP to avoid 1h timeout
    logging_steps=50,
    save_strategy="no",
    evaluation_strategy="no",
    report_to="none"
)


# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

print("🚀 Starting training...")
trainer.train()
print("✅ Training completed")

# -------------------------------
# SAVE MODEL ARTIFACTS (CRITICAL)
# -------------------------------
print("💾 Saving HuggingFace model artifacts...")

os.makedirs(MODEL_DIR, exist_ok=True)

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Model saved to /opt/ml/model")
print("🎯 SageMaker will package this into model.tar.gz automatically")
