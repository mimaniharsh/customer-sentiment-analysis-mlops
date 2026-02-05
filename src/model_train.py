import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
from io import StringIO
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

def train_model(bucket_name, project_name, aws_region):
    """Train DistilBERT model on CPU"""
    
    # Force CPU usage
    device = torch.device('cpu')
    print(f"🖥️  Using device: {device}")
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download preprocessed data with project path
    print("📥 Downloading training data...")
    train_key = f"{project_name}/data/processed/train.csv"
    val_key = f"{project_name}/data/processed/val.csv"
    
    train_obj = s3_client.get_object(Bucket=bucket_name, Key=train_key)
    train = pd.read_csv(train_obj['Body'])
    
    val_obj = s3_client.get_object(Bucket=bucket_name, Key=val_key)
    val = pd.read_csv(val_obj['Body'])
    
    print(f"✅ Train: {len(train)}, Val: {len(val)}")
    
    # Tokenize
    print("🔤 Tokenizing...")
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
    
    # Create datasets
    train_dataset = TweetDataset(train_encodings, train['label'].tolist())
    val_dataset = TweetDataset(val_encodings, val['label'].tolist())
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1, 2]),
        y=train['label'].tolist()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"⚖️  Class weights: {class_weights}")
    
    # Load model and move to CPU
    print("🤖 Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none',
        no_cuda=True,  # Force CPU training
        use_cpu=True
    )
    
    # Create trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("🚀 Training model on CPU...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
    
    # Upload to S3 with project path
    print("📤 Uploading model to S3...")
    for root, dirs, files in os.walk('./model'):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = f"{project_name}/models/distilbert/{file}"
            s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f"   ✅ {file}")
    
    print("✅ Training complete!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBERT model')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    
    args = parser.parse_args()
    
    sys.exit(train_model(args.bucket, args.project, args.region))
