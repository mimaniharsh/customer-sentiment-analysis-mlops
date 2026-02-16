#!/usr/bin/env python3
"""
Model Evaluation Script
Works with models saved by SageMaker training jobs
"""
import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
import tarfile
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

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

def find_latest_model(s3_client, bucket_name, project_name):
    """Find latest SageMaker training model"""
    
    print("🔍 Searching for trained model in S3...")
    prefix = f"{project_name}/models/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            # Find model.tar.gz files
            model_files = [
                obj for obj in response['Contents'] 
                if obj['Key'].endswith('model.tar.gz') and '/output/' in obj['Key']
            ]
            
            if model_files:
                latest = max(model_files, key=lambda x: x['LastModified'])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found model: {model_uri}")
                print(f"   Last modified: {latest['LastModified']}")
                return latest['Key']
    except Exception as e:
        print(f"❌ Error searching S3: {e}")
    
    raise FileNotFoundError(
        f"No trained model found in s3://{bucket_name}/{project_name}/models/.\n"
        "Please run training workflow first."
    )

def download_and_extract_model(s3_client, bucket_name, model_key):
    """Download and extract model from S3"""
    
    print(f"\n📥 Downloading model from S3...")
    os.makedirs('./model', exist_ok=True)
    
    # Download model.tar.gz
    s3_client.download_file(bucket_name, model_key, 'model.tar.gz')
    print(f"   ✅ model.tar.gz downloaded")
    
    # Extract
    print("📂 Extracting model files...")
    with tarfile.open('model.tar.gz', 'r:gz') as tar:
        tar.extractall('./model')
    
    # List extracted files
    files = os.listdir('./model')
    print(f"   Extracted {len(files)} files:")
    for f in files:
        size = os.path.getsize(f'./model/{f}')
        print(f"   ✅ {f} ({size:,} bytes)")
    
    os.remove('model.tar.gz')
    
    # Verify critical files exist
    required_files = ['config.json']
    has_weights = any(f in files for f in ['model.safetensors', 'pytorch_model.bin'])
    
    if not all(f in files for f in required_files) or not has_weights:
        raise FileNotFoundError(f"Missing required model files.\nFound: {files}")
    
    return True

def evaluate_model(args):
    """Evaluate trained model on test set"""
    
    print("="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    s3_client = boto3.client('s3', region_name=args.region)
    
    # Find and download model
    model_key = find_latest_model(s3_client, args.bucket, args.project)
    download_and_extract_model(s3_client, args.bucket, model_key)
    
    # Download test data
    print(f"\n📥 Downloading test data...")
    test_key = f"{args.project}/data/processed/test.csv"
    test_obj = s3_client.get_object(Bucket=args.bucket, Key=test_key)
    test = pd.read_csv(test_obj['Body'])
    print(f"✅ Test set: {len(test)} samples")
    
    # Load model and tokenizer
    print("\n🤖 Loading model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('./model')
    model = DistilBertForSequenceClassification.from_pretrained('./model')
    model.eval()
    
    # UPDATED: Dynamic device mapping (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"✅ Model loaded successfully on {device}")
    
    # Tokenize test data
    print("\n🔤 Tokenizing test data...")
    test_encodings = tokenizer(
        test['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    test_dataset = TweetDataset(test_encodings, test['label'].tolist())
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = []
    true_labels = test['label'].tolist()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if i > 0 and i % 500 == 0:
                print(f"   Processed {i}/{len(test_dataset)} samples...")
            
            item = test_dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    print(f"\n✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['negative', 'neutral', 'positive']))
    
    cm = confusion_matrix(true_labels, predictions)
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': classification_report(
            true_labels, predictions, 
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save metrics JSON
    metrics_json = json.dumps(metrics, indent=2)
    with open(f'{args.output_path}/evaluation_results.json', 'w') as f:
        f.write(metrics_json)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix - Sentiment Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{args.output_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Upload results to S3
    metrics_key = f"{args.project}/evaluation/evaluation_results.json"
    cm_key = f"{args.project}/evaluation/confusion_matrix.png"
    
    s3_client.put_object(Bucket=args.bucket, Key=metrics_key, Body=metrics_json)
    s3_client.upload_file(f'{args.output_path}/confusion_matrix.png', args.bucket, cm_key)
    
    print(f"\n✅ Metrics saved to s3://{args.bucket}/{metrics_key}")
    print(f"✅ Confusion matrix saved to s3://{args.bucket}/{cm_key}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', required=True, help='AWS region')
    parser.add_argument('--output-path', default='/tmp/evaluation', help='Local output path')
    
    args = parser.parse_args()
    
    try:
        sys.exit(evaluate_model(args))
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)