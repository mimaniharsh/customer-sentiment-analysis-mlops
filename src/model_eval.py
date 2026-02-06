import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import tarfile

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Make matplotlib CI-safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Dataset
# -------------------------------
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


# -------------------------------
# Evaluation logic
# -------------------------------
def evaluate_model(bucket, project, region, model_uri, output_path):
    print("=" * 70)
    print("📊 MODEL EVALUATION")
    print("=" * 70)

    s3 = boto3.client("s3", region_name=region)

    # -------------------------------
    # Load test data
    # -------------------------------
    print("📥 Downloading test data...")
    test_key = f"{project}/data/processed/test.csv"
    obj = s3.get_object(Bucket=bucket, Key=test_key)
    test_df = pd.read_csv(obj["Body"])
    print(f"✅ Test set loaded: {len(test_df)} samples")

    # -------------------------------
    # Download EXACT model artifact
    # -------------------------------
    print(f"\n📦 Using model artifact:\n{model_uri}")

    model_bucket = model_uri.replace("s3://", "").split("/", 1)[0]
    model_key = model_uri.replace(f"s3://{model_bucket}/", "")

    s3.download_file(model_bucket, model_key, "model.tar.gz")

    print("📂 Extracting model.tar.gz...")
    with tarfile.open("model.tar.gz", "r:gz") as tar:
        tar.extractall("model")

    # -------------------------------
    # Locate HF model directory
    # -------------------------------
    model_root = Path("model")
    hf_dir = None

    for p in model_root.rglob("*"):
        if p.is_dir() and (
            (p / "pytorch_model.bin").exists()
            or (p / "model.safetensors").exists()
        ):
            hf_dir = p
            break

    if hf_dir is None:
        raise FileNotFoundError("❌ HuggingFace model weights not found")

    print(f"✅ Found model weights in: {hf_dir}")

    # -------------------------------
    # Load model
    # -------------------------------
    print("🤖 Loading model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(hf_dir)
    model = DistilBertForSequenceClassification.from_pretrained(hf_dir)
    model.eval()

    # -------------------------------
    # Tokenize test data
    # -------------------------------
    encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    dataset = TweetDataset(encodings, test_df["label"].tolist())

    # -------------------------------
    # Inference
    # -------------------------------
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            inputs = {k: v.unsqueeze(0) for k, v in item.items() if k != "labels"}
            outputs = model(**inputs)
            preds.append(outputs.logits.argmax(-1).item())

    y_true = test_df["label"].tolist()

    # -------------------------------
    # Metrics
    # -------------------------------
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="weighted"
    )

    print("\n📊 RESULTS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print(classification_report(y_true, preds))

    cm = confusion_matrix(y_true, preds)

    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist()
            },
            f,
            indent=2
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_path}/confusion_matrix.png")
    plt.close()

    print("\n✅ Evaluation complete")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--output-path", default="/tmp/evaluation")

    args = parser.parse_args()

    evaluate_model(
        args.bucket,
        args.project,
        args.region,
        args.model_uri,
        args.output_path
    )
