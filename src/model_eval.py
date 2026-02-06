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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
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

def evaluate_model(bucket_name, project_name, aws_region, output_path):
    """Evaluate trained model on test set"""
    
    print("="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
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
    probabilities = []
    true_labels = test['label'].tolist()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy()[0])
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['negative', 'neutral', 'positive']))
    
    cm = confusion_matrix(true_labels, predictions)
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    # Save metrics with project path
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
    os.makedirs(output_path, exist_ok=True)
    
    # Save metrics JSON
    metrics_json = json.dumps(metrics, indent=2)
    with open(f'{output_path}/evaluation_results.json', 'w') as f:
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
    plt.savefig(f'{output_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Upload results to S3
    metrics_key = f"{project_name}/evaluation/evaluation_results.json"
    cm_key = f"{project_name}/evaluation/confusion_matrix.png"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metrics_key,
        Body=metrics_json
    )
    s3_client.upload_file(
        f'{output_path}/confusion_matrix.png',
        bucket_name,
        cm_key
    )
    
    print(f"\n✅ Metrics saved to s3://{bucket_name}/{metrics_key}")
    print(f"✅ Confusion matrix saved to s3://{bucket_name}/{cm_key}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    parser.add_argument('--output-path', default='/tmp/evaluation', help='Local output path')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(args.bucket, args.project, args.region, args.output_path)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
