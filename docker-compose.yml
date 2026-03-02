#!/usr/bin/env python3
"""
packages/ml/evaluation/model_eval.py  — v2 (upgraded)

FIXES vs original:
  FIX 1 — Batched prediction via DataLoader (was a Python for-loop, 1 sample at a time)
           10-50x faster on large test sets.
  FIX 2 — Expected Calibration Error (ECE) computed  (was missing)
           Measures how well-calibrated confidence scores are (lower = better).
  FIX 3 — Dynamic device: uses GPU when available (was always CPU)
  FIX 4 — Reads max_length from sentient_config.json saved at training time
"""
from __future__ import annotations
import os, sys, json, argparse, tarfile
import boto3
import pandas as pd
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    FIX 2: ECE — measures calibration quality.
    A well-calibrated model with confidence=0.8 should be correct ~80% of the time.
    ECE < 0.05 is excellent; ECE > 0.15 suggests significant overconfidence.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def find_and_download_model(s3, bucket: str, project: str) -> bool:
    prefix = f"{project}/models/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [
        o for o in resp.get("Contents", [])
        if o["Key"].endswith("model.tar.gz") and "/output/" in o["Key"]
    ]
    if not files:
        raise FileNotFoundError(f"No model found at s3://{bucket}/{prefix}")
    latest = max(files, key=lambda x: x["LastModified"])
    print(f"  Model: s3://{bucket}/{latest['Key']}")
    s3.download_file(bucket, latest["Key"], "model.tar.gz")
    os.makedirs("./model", exist_ok=True)
    with tarfile.open("model.tar.gz", "r:gz") as tar:
        tar.extractall("./model")
    os.remove("model.tar.gz")
    return True


def evaluate_model(args) -> int:
    print("=" * 70)
    print("MODEL EVALUATION v2")
    print("=" * 70)

    s3 = boto3.client("s3", region_name=args.region)
    find_and_download_model(s3, args.bucket, args.project)

    # FIX 3: dynamic device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # FIX 4: read max_length from training config
    config_path = "./model/sentient_config.json"
    max_length = 128
    if os.path.exists(config_path):
        with open(config_path) as f:
            max_length = json.load(f).get("max_length", 128)
    print(f"Token max_length: {max_length}  (from training config)")

    tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
    model = DistilBertForSequenceClassification.from_pretrained("./model")
    model.eval()
    model.to(device)

    # Load test set
    test_obj = s3.get_object(Bucket=args.bucket, Key=f"{args.project}/data/processed/test.csv")
    test_df = pd.read_csv(test_obj["Body"])
    print(f"Test set: {len(test_df):,} samples")

    enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=max_length)
    dataset = EvalDataset(enc, test_df["label"].tolist())

    # FIX 1: batched prediction via DataLoader (was per-sample loop)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"\nRunning batched inference (batch={args.batch_size})...")

    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            labels = batch.pop("labels")
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1).cpu()
            preds  = probs.argmax(dim=-1)
            confs  = probs.max(dim=-1).values

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.numpy())

            if (batch_idx + 1) % 20 == 0:
                done = (batch_idx + 1) * args.batch_size
                print(f"  {done:,}/{len(dataset):,}")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)
    correct    = (all_preds == all_labels).astype(float)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    f1_per = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    ece = expected_calibration_error(all_confs, correct)  # FIX 2

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  F1 (weighted)  : {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  ECE            : {ece:.4f}  (< 0.05 = excellent, > 0.15 = needs calibration)")
    print(f"  F1 negative    : {f1_per[0]:.4f}")
    print(f"  F1 neutral     : {f1_per[1]:.4f}  <- typically weakest")
    print(f"  F1 positive    : {f1_per[2]:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=["negative","neutral","positive"]))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)

    os.makedirs(args.output_path, exist_ok=True)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["negative","neutral","positive"],
                yticklabels=["negative","neutral","positive"])
    plt.title("Confusion Matrix — Sentiment Classification", fontsize=14, fontweight="bold")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/confusion_matrix.png", dpi=150)
    plt.close()

    # Calibration reliability diagram
    bin_edges = np.linspace(0, 1, 16)
    bin_accs, bin_confs_avg = [], []
    for i in range(15):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (all_confs > lo) & (all_confs <= hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs_avg.append(all_confs[mask].mean())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(bin_confs_avg, bin_accs, "o-", color="steelblue", label=f"Model (ECE={ece:.3f})")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/calibration_plot.png", dpi=150)
    plt.close()

    metrics = {
        "accuracy":  float(accuracy),
        "f1_weighted": float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "ece":       float(ece),
        "f1_negative": float(f1_per[0]),
        "f1_neutral":  float(f1_per[1]),
        "f1_positive": float(f1_per[2]),
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=["negative","neutral","positive"], output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "n_test": len(all_labels),
    }

    metrics_json = json.dumps(metrics, indent=2)
    with open(f"{args.output_path}/evaluation_results.json", "w") as f:
        f.write(metrics_json)

    # Upload to S3
    s3.put_object(Bucket=args.bucket, Key=f"{args.project}/evaluation/evaluation_results.json", Body=metrics_json)
    s3.upload_file(f"{args.output_path}/confusion_matrix.png",  args.bucket, f"{args.project}/evaluation/confusion_matrix.png")
    s3.upload_file(f"{args.output_path}/calibration_plot.png",  args.bucket, f"{args.project}/evaluation/calibration_plot.png")

    print(f"\nResults uploaded to s3://{args.bucket}/{args.project}/evaluation/")
    print("EVALUATION COMPLETE")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",      required=True)
    parser.add_argument("--project",     required=True)
    parser.add_argument("--region",      required=True)
    parser.add_argument("--output-path", default="/tmp/evaluation")
    parser.add_argument("--batch-size",  type=int, default=64, help="Inference batch size for DataLoader")
    args = parser.parse_args()
    try:
        sys.exit(evaluate_model(args))
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
