import os
import sys
import json
import argparse
import boto3
import pandas as pd
import torch
import numpy as np
from io import StringIO
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

def evaluate_model(bucket_name, project_name, aws_region):
    """Evaluate trained model on test set"""
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download test data with project path
    print("📥 Downloading test data...")
    test_key = f"{project_name}/data/processed/test.csv"
    test_obj = s3_client.get_object(Bucket=bucket_name, Key=test_key)
    test = pd.read_csv(test_obj['Body'])
    print(f"✅ Test set: {len(test)} samples")
    
    # Download model with project path
    print("📥 Downloading model from S3...")
    os.makedirs('./model', exist_ok=True)
    
    model_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 
                   'vocab.txt', 'special_tokens_map.json', 'tokenizer.json']
    
    for file in model_files:
        try:
            s3_key = f"{project_name}/models/distilbert/{file}"
            s3_client.download_file(bucket_name, s3_key, f'./model/{file}')
            print(f"   ✅ {file}")
        except Exception as e:
            print(f"   ⚠️  Could not download {file}: {e}")
    
    # Load model and tokenizer
    print("🤖 Loading model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('./model')
    model = DistilBertForSequenceClassification.from_pretrained('./model')
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    # Tokenize test data
    print("🔤 Tokenizing test data...")
    test_encodings = tokenizer(
        test['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    test_dataset = TweetDataset(test_encodings, test['label'].tolist())
    
    # Predict
    print("🔮 Making predictions...")
    predictions = []
    true_labels = test['label'].tolist()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['negative', 'neutral', 'positive']))
    
    print("📊 Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # Save metrics with project path
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': classification_report(
            true_labels, predictions, 
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
    }
    
    metrics_json = json.dumps(metrics, indent=2)
    metrics_key = f"{project_name}/models/distilbert/metrics.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metrics_key,
        Body=metrics_json
    )
    print(f"\n✅ Metrics saved to s3://{bucket_name}/{metrics_key}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    
    args = parser.parse_args()
    
    sys.exit(evaluate_model(args.bucket, args.project, args.region))
