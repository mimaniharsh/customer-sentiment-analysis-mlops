import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from io import StringIO
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

def find_model_directory(root: Path) -> Path:
    """
    Recursively search for a directory containing HuggingFace model weights.
    """
    for path in root.rglob("*"):
        if path.is_dir():
            if (
                (path / "pytorch_model.bin").exists()
                or (path / "model.safetensors").exists()
            ):
                return path
    return None

# Make matplotlib CI-safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile


# -------------------------------
# Dataset wrapper
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
# Main evaluation function
# -------------------------------
def evaluate_model(bucket_name, project_name, aws_region, output_path):
    print("=" * 70)
    print("📊 MODEL EVALUATION")
    print("=" * 70)

    s3_client = boto3.client("s3", region_name=aws_region)

    # -------------------------------
    # Load test data
    # -------------------------------
    print("📥 Downloading test data...")
    test_key = f"{project_name}/data/processed/test.csv"
    test_obj = s3_client.get_object(Bucket=bucket_name, Key=test_key)
    test_df = pd.read_csv(test_obj["Body"])

    print(f"✅ Test set loaded: {len(test_df)} samples")

    # -------------------------------
    # Download model (robust logic)
    # -------------------------------
    print("\n📥 Downloading model from S3...")
    model_root = Path("./model")
    model_root.mkdir(exist_ok=True)

    found_model_dir = None

    # ---- Option 1: Direct HuggingFace-style upload
    direct_prefix = f"{project_name}/models/distilbert/"
    print(f"🔍 Checking direct upload: s3://{bucket_name}/{direct_prefix}")

    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=direct_prefix
    )

    if "Contents" in response:
        weight_keys = [
            obj["Key"] for obj in response["Contents"]
            if obj["Key"].endswith(("pytorch_model.bin", "model.safetensors"))
        ]

        if weight_keys:
            print("✅ Found direct HuggingFace model files")
            for obj in response["Contents"]:
                filename = obj["Key"].split("/")[-1]
                if filename:
                    s3_client.download_file(
                        bucket_name,
                        obj["Key"],
                        model_root / filename
                    )
            found_model_dir = model_root

    # ---- Option 2: SageMaker training output
    if found_model_dir is None:
        print("🔍 Checking SageMaker training outputs...")
        sm_prefix = f"{project_name}/models/"
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=sm_prefix
        )

        tar_key = None
        for obj in response.get("Contents", []):
            if obj["Key"].endswith("model.tar.gz"):
                tar_key = obj["Key"]
                break

        if tar_key is None:
            raise FileNotFoundError(
                "❌ No valid model found (neither direct upload nor SageMaker output)."
            )

        print(f"📦 Downloading SageMaker model archive: {tar_key}")
        s3_client.download_file(bucket_name, tar_key, "model.tar.gz")

        print("📂 Extracting model.tar.gz...")
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(model_root)

        # SageMaker usually nests model files inside ./model/model/
        if (model_root / "model").exists():
            found_model_dir = model_root / "model"
        else:
            found_model_dir = model_root

    # -------------------------------
    # Validate model files
    # -------------------------------
print("🔎 Searching for model weights recursively...")
actual_model_dir = find_model_directory(found_model_dir)

if actual_model_dir is None:
    raise FileNotFoundError(
        f"❌ Could not locate HuggingFace model weights inside {found_model_dir}"
    )

print(f"✅ Found model weights in: {actual_model_dir}")

    for rf in required_files:
        if not (found_model_dir / rf).exists():
            raise FileNotFoundError(
                f"❌ Required file missing: {rf} in {found_model_dir}"
            )

    # -------------------------------
    # Load model
    # -------------------------------
    print(f"\n🤖 Loading model from: {found_model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(actual_model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(actual_model_dir)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # -------------------------------
    # Tokenize test data
    # -------------------------------
    print("🔤 Tokenizing test data...")
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
    print("🔮 Running inference...")
    predictions = []
    true_labels = test_df["label"].tolist()

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)

    # -------------------------------
    # Metrics
    # -------------------------------
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )

    print(f"✅ Accuracy : {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall   : {recall:.4f}")
    print(f"✅ F1-score : {f1:.4f}\n")

    print("📋 Classification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=["negative", "neutral", "positive"]
    ))

    cm = confusion_matrix(true_labels, predictions)

    # -------------------------------
    # Save artifacts
    # -------------------------------
    os.makedirs(output_path, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

    with open(f"{output_path}/evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_path}/confusion_matrix.png", dpi=300)
    plt.close()

    # -------------------------------
    # Upload results to S3
    # -------------------------------
    s3_client.put_object(
        Bucket=bucket_name,
        Key=f"{project_name}/evaluation/evaluation_results.json",
        Body=json.dumps(metrics, indent=2)
    )

    s3_client.upload_file(
        f"{output_path}/confusion_matrix.png",
        bucket_name,
        f"{project_name}/evaluation/confusion_matrix.png"
    )

    print("\n✅ Evaluation complete and artifacts uploaded.")
    print("=" * 70)


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--output-path", default="/tmp/evaluation")

    args = parser.parse_args()

    try:
        evaluate_model(
            args.bucket,
            args.project,
            args.region,
            args.output_path
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
