#!/usr/bin/env python3
"""
packages/ml/training/model_train.py  — v2 (upgraded)

FIXES vs original:
  1. warmup_ratio=0.06  (was hardcoded warmup_steps=500 — doesn't scale)
  2. DataCollatorWithPadding — dynamic per-batch padding (was padding entire dataset)
  3. lr_scheduler_type='cosine'  (was default linear)
  4. CrossEntropyLoss cached in WeightedTrainer  (was re-instantiated every fwd pass)
  5. label_smoothing_factor=0.1  (prevents overconfidence, improves calibration)
  6. EarlyStoppingCallback  (stops before overfitting)
  7. Per-class F1 logged  (see neutral class weakness clearly)
  8. group_by_length=True  (minimises padding waste per batch)
  9. gradient_accumulation_steps param  (larger effective batch on constrained GPU)
 10. --max-length arg  (128 for tweets, set 256 for tickets/reviews/long feedback)
"""
from __future__ import annotations
import os, sys, json, argparse
import boto3
import pandas as pd
import torch
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    """Weighted cross-entropy; loss_fct cached (not re-created per forward pass)."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._loss_fct = None  # lazily built on first call once device is known

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self._loss_fct is None:
            self._loss_fct = torch.nn.CrossEntropyLoss(
                weight=self._class_weights.to(outputs.logits.device)
            )
        loss = self._loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_negative": round(f1_per[0], 4),
        "f1_neutral":  round(f1_per[1], 4),   # typically the weakest — watch this
        "f1_positive": round(f1_per[2], 4),
    }


def train_model(args) -> int:
    print("=" * 70)
    print("SAGEMAKER TRAINING — SENTIMENT ANALYSIS v2")
    print("=" * 70)
    print(f"  Batch size   : {args.batch_size}  x accum {args.grad_accum}  =  eff. {args.batch_size*args.grad_accum}")
    print(f"  Max epochs   : {args.epochs}  (early-stop patience={args.patience})")
    print(f"  Max length   : {args.max_length} tokens")
    print(f"  LR scheduler : cosine  |  label_smooth=0.1  |  warmup_ratio=0.06")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    s3 = boto3.client("s3", region_name=args.region)

    def _load(key):
        return pd.read_csv(s3.get_object(Bucket=args.bucket, Key=key)["Body"])

    print("\nDownloading data...")
    train_df = _load(f"{args.project}/data/processed/train.csv")
    val_df   = _load(f"{args.project}/data/processed/val.csv")
    print(f"  Train: {len(train_df):,}   Val: {len(val_df):,}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def _tok(texts):
        return tokenizer(texts, truncation=True, padding=False, max_length=args.max_length)

    print("Tokenising (no static padding — DataCollator handles it per batch)...")
    train_enc = _tok(train_df["text"].tolist())
    val_enc   = _tok(val_df["text"].tolist())

    train_dataset = SentimentDataset(train_enc, train_df["label"].tolist())
    val_dataset   = SentimentDataset(val_enc,   val_df["label"].tolist())

    print("Computing class weights...")
    weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=train_df["label"].tolist())
    class_weights = torch.tensor(weights, dtype=torch.float)
    for i, (name, w) in enumerate(zip(["negative","neutral","positive"], weights)):
        print(f"  {name:10s}: {w:.3f}")

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(device)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.06,                          # FIX 1: scales with dataset
        weight_decay=0.01,
        lr_scheduler_type="cosine",                 # FIX 3: better than linear
        label_smoothing_factor=0.1,                 # FIX 5: calibration
        logging_dir="/opt/ml/output/tensorboard",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        no_cuda=(device.type == "cpu"),
        fp16=(device.type == "cuda"),
        save_total_limit=2,
        group_by_length=True,                       # FIX 8: less padding waste
        dataloader_num_workers=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # FIX 2
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],  # FIX 6
    )

    print("\nTraining...")
    train_result = trainer.train()
    eval_result  = trainer.evaluate()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Val accuracy  : {eval_result.get('eval_accuracy',0):.4f}")
    print(f"  Val F1 (wtd)  : {eval_result.get('eval_f1',0):.4f}")
    print(f"  Val F1 neg    : {eval_result.get('eval_f1_negative',0):.4f}")
    print(f"  Val F1 neu    : {eval_result.get('eval_f1_neutral',0):.4f}  <- typically weakest")
    print(f"  Val F1 pos    : {eval_result.get('eval_f1_positive',0):.4f}")
    print("=" * 70)

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save max_length for inference script
    with open(os.path.join(model_dir, "sentient_config.json"), "w") as f:
        json.dump({"max_length": args.max_length, "label_map": {"0":"negative","1":"neutral","2":"positive"}}, f, indent=2)

    metrics = {
        "train_loss": float(train_result.training_loss),
        "eval_accuracy": float(eval_result.get("eval_accuracy", 0)),
        "eval_f1_weighted": float(eval_result.get("eval_f1", 0)),
        "eval_f1_negative": float(eval_result.get("eval_f1_negative", 0)),
        "eval_f1_neutral":  float(eval_result.get("eval_f1_neutral", 0)),
        "eval_f1_positive": float(eval_result.get("eval_f1_positive", 0)),
        "hyperparameters": {
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "max_length": args.max_length, "lr_scheduler": "cosine",
            "label_smoothing": 0.1, "warmup_ratio": 0.06,
        },
    }
    with open(os.path.join(model_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved files:")
    for fname in sorted(os.listdir(model_dir)):
        print(f"  {fname}  ({os.path.getsize(os.path.join(model_dir, fname)):,} bytes)")

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",     required=True)
    parser.add_argument("--project",    required=True)
    parser.add_argument("--region",     required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--max-length", type=int, default=128,
                        help="128 for tweets; use 256 for tickets/reviews/long feedback")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation. Effective batch = batch_size x grad_accum")
    parser.add_argument("--patience",   type=int, default=3,
                        help="Early stopping patience in epochs")
    args = parser.parse_args()
    try:
        sys.exit(train_model(args))
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
