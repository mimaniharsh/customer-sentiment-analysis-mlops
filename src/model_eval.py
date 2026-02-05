import os
import sys
import json
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

def evaluate_model():
    """Evaluate trained model on test set"""
    
    bucket_name = os.environ['S3_BUCKET']
    aws_region = os.environ['AWS_REGION']
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download test data
    print("Downloading test data...")
    test_obj = s3_client.get_object(Bucket=bucket_name, Key='data/preprocessed/test.csv')
    test = pd.read_csv(test_obj['Body'])
    print(f"Test set: {len(test)} samples")
    
    # Download model
    print("Downloading model from S3...")
    os.makedirs('./model', exist_ok=True)
    
    model_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 
                   'vocab.txt', 'special_tokens_map.json', 'tokenizer.json']
    
    for file in model_files:
        try:
            s3_client.download_file(bucket_name, f'models/distilbert/{file}', f'./model/{file}')
        except Exception as e:
            print(f"Warning: Could not download {file}: {e}")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('./model')
    model = DistilBertForSequenceClassification.from_pretrained('./model')
    model.eval()
    
    # Tokenize test data
    print("Tokenizing test data...")
    test_encodings = tokenizer(
        test['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    test_dataset = TweetDataset(test_encodings, test['label'].tolist())
    
    # Predict
    print("Making predictions...")
    predictions = []
    true_labels = test['label'].tolist()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            inputs = {k: v.unsqueeze(0) for k, v in item.items() if k != 'labels'}
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)
    
    # Calculate metrics
    print("\n=== Evaluation Results ===")
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['negative', 'neutral', 'positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': classification_report(
            true_labels, predictions, 
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
    }
    
    metrics_json = json.dumps(metrics, indent=2)
    s3_client.put_object(
        Bucket=bucket_name,
        Key='models/distilbert/metrics.json',
        Body=metrics_json
    )
    print("\nMetrics saved to S3")
    
    return 0

if __name__ == "__main__":
    sys.exit(evaluate_model())
