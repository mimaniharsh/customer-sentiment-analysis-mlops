"""
SageMaker-compatible HuggingFace training script
Customer Sentiment Analysis using DistilBERT
"""

import os
import argparse
import pandas as pd
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# --------------------------------------------------
# Argument parsing (from SageMaker hyperparameters)
# --------------------------------------------------
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
print(f"EPOCHS     : {EPOCHS}")
print(f"BATCH_SIZE : {BATCH_SIZE}")
print("=====================================")

# --------------------------------------------------
# SageMaker paths (CRITICAL)
# --------------------------------------------------
TRAINING_DATA_DIR = "/opt/ml/input/data/training"
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

print(f"Training data dir : {TRAINING_DATA_DIR}")
print(f"Model output dir  : {MODEL_DIR}")

# --------------------------------------------------
# Load training data
# --------------------------------------------------
print("📥 Loading training data...")

files = os.listdir(TRAINING_DATA_DIR)
if not files:
    raise RuntimeError("❌ No training data found in SageMaker input channel")

data_path = os.path.join(TRAINING_DATA_DIR, files[0])
df = pd.read_csv(data_path)

if not {"text", "label"}.issubset(df.columns):
    raise ValueError("❌ Dataset must contain 'text' and 'label' columns")

print(f"✅ Dataset loaded with shape: {df.shape}")

# --------------------------------------------------
# Train / validation split
# --------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
print("🔤 Tokenizing text...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

train_encodings = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TweetDataset(train_encodings, train_df["label"].tolist())
val_dataset = TweetDataset(val_encodings, val_df["label"].tolist())

# --------------------------------------------------
# Load model
# --------------------------------------------------
print("🤖 Loading DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# --------------------------------------------------
# Training arguments (CPU-safe for SageMaker)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="/tmp/results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    report_to="none",
    no_cuda=True
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# --------------------------------------------------
# Train
# --------------------------------------------------
print("🚀 Starting training...")
trainer.train()

# --------------------------------------------------
# Save model to SageMaker model directory (MANDATORY)
# --------------------------------------------------
print(f"💾 Saving model to: {MODEL_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ Model and tokenizer saved successfully")
print("🎉 Training job completed")
