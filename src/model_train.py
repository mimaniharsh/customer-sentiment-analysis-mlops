#!/usr/bin/env python3
"""
SageMaker Training Script for Sentiment Analysis
Saves model to /opt/ml/model for SageMaker compatibility
"""
import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_model(args):
    """Train DistilBERT model for SageMaker"""
    
    print("="*70)
    print("🚀 SAGEMAKER TRAINING - SENTIMENT ANALYSIS")
    print("="*70)
    print(f"📦 Bucket: {args.bucket}")
    print(f"📁 Project: {args.project}")
    print(f"🌍 Region: {args.region}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print("="*70)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using CPU")
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=args.region)
    
    # Download preprocessed data
    print("\n📥 Downloading training data from S3...")
    train_key = f"{args.project}/data/processed/train.csv"
    val_key = f"{args.project}/data/processed/val.csv"
    
    try:
        train_obj = s3_client.get_object(Bucket=args.bucket, Key=train_key)
        train = pd.read_csv(train_obj['Body'])
        print(f"   ✅ Train: {len(train)} samples")
    except Exception as e:
        print(f"   ❌ Failed to download train data from s3://{args.bucket}/{train_key}")
        print(f"      Error: {e}")
        raise
    
    try:
        val_obj = s3_client.get_object(Bucket=args.bucket, Key=val_key)
        val = pd.read_csv(val_obj['Body'])
        print(f"   ✅ Val: {len(val)} samples")
    except Exception as e:
        print(f"   ❌ Failed to download val data from s3://{args.bucket}/{val_key}")
        print(f"      Error: {e}")
        raise
    
    # Tokenize
    print("\n🔤 Tokenizing text...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(
        train['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    val_encodings = tokenizer(
        val['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    print("   ✅ Tokenization complete")
    
    # Create datasets
    train_dataset = TweetDataset(train_encodings, train['label'].tolist())
    val_dataset = TweetDataset(val_encodings, val['label'].tolist())
    
    # Calculate class weights
    print("\n⚖️  Calculating class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1, 2]),
        y=train['label'].tolist()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"   Negative: {class_weights[0]:.3f}")
    print(f"   Neutral:  {class_weights[1]:.3f}")
    print(f"   Positive: {class_weights[2]:.3f}")
    
    # Load model
    print("\n🤖 Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    model.to(device)
    print("   ✅ Model loaded")
    
    # CRITICAL: Use SageMaker's model directory
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    print(f"\n💾 Model will be saved to: {model_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='/opt/ml/output/tensorboard',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none',
        no_cuda=(device.type == 'cpu'),
        fp16=(device.type == 'cuda'),
        save_total_limit=1  # Only keep best model
    )
    
    # Create trainer
    print("\n🏋️  Creating trainer...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("="*70)
    train_result = trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating on validation set...")
    eval_result = trainer.evaluate()
    
    print("\n" + "="*70)
    print("📈 TRAINING RESULTS")
    print("="*70)
    print(f"Train loss:     {train_result.training_loss:.4f}")
    print(f"Val accuracy:   {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"Val F1:         {eval_result.get('eval_f1', 0):.4f}")
    print(f"Val precision:  {eval_result.get('eval_precision', 0):.4f}")
    print(f"Val recall:     {eval_result.get('eval_recall', 0):.4f}")
    print("="*70)
    
    # CRITICAL: Save model and tokenizer to SageMaker directory
    print(f"\n💾 Saving model and tokenizer to {model_dir}...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Verify files were saved
    saved_files = os.listdir(model_dir)
    print(f"\n✅ Saved files in {model_dir}:")
    for f in saved_files:
        file_size = os.path.getsize(os.path.join(model_dir, f))
        print(f"   {f} ({file_size:,} bytes)")
    
    # Save training metrics
    metrics = {
        'train_loss': float(train_result.training_loss),
        'train_runtime': train_result.metrics['train_runtime'],
        'eval_accuracy': float(eval_result.get('eval_accuracy', 0)),
        'eval_f1': float(eval_result.get('eval_f1', 0)),
        'eval_precision': float(eval_result.get('eval_precision', 0)),
        'eval_recall': float(eval_result.get('eval_recall', 0)),
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
    }
    
    metrics_file = os.path.join(model_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ training_metrics.json")
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {model_dir}")
    print("   SageMaker will upload this to S3 automatically")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBERT for sentiment analysis')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--project', type=str, required=True, help='Project name/prefix')
    parser.add_argument('--region', type=str, required=True, help='AWS region')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    args = parser.parse_args()
    
    try:
        sys.exit(train_model(args))
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
