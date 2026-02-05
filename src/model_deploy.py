import os
import sys
import json
import argparse
import boto3
import tarfile
from datetime import datetime

def create_model_tar(bucket_name, project_name, aws_region):
    """Create model.tar.gz for SageMaker"""
    print("📦 Creating model.tar.gz...")
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download model files with project path
    os.makedirs('./model', exist_ok=True)
    
    model_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 
                   'vocab.txt', 'special_tokens_map.json', 'tokenizer.json']
    
    print("📥 Downloading model files...")
    for file in model_files:
        s3_key = f"{project_name}/models/distilbert/{file}"
        s3_client.download_file(bucket_name, s3_key, f'./model/{file}')
        print(f"   ✅ {file}")
    
    # Create inference script
    inference_code = '''import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def model_fn(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    return {'model': model, 'tokenizer': tokenizer, 'device': device}

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data['text']
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    
    inputs = tokenizer(input_data, truncation=True, padding=True, 
                      max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = outputs.logits.argmax(-1).item()
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    return {
        'prediction': label_map[pred],
        'confidence': float(probs[0][pred]),
        'probabilities': {
            'negative': float(probs[0][0]),
            'neutral': float(probs[0][1]),
            'positive': float(probs[0][2])
        }
    }

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
'''
    
    with open('./model/inference.py', 'w') as f:
        f.write(inference_code)
    print("   ✅ inference.py")
    
    # Create requirements.txt for inference
    requirements = '''torch==2.3.1
transformers==4.44.2
'''
    with open('./model/requirements.txt', 'w') as f:
        f.write(requirements)
    print("   ✅ requirements.txt")
    
    # Create tar.gz
    print("🗜️  Compressing model...")
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    # Upload to S3 with project path
    s3_key = f"{project_name}/models/distilbert/model.tar.gz"
    print(f"📤 Uploading model.tar.gz to s3://{bucket_name}/{s3_key}")
    s3_client.upload_file('model.tar.gz', bucket_name, s3_key)
    
    return f"s3://{bucket_name}/{s3_key}"

def deploy_model(bucket_name, project_name, endpoint_name, sagemaker_role, aws_region):
    """Deploy model to SageMaker endpoint"""
    
    # Create model artifact
    model_data = create_model_tar(bucket_name, project_name, aws_region)
    
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{project_name}-model-{timestamp}"
    endpoint_config_name = f"{project_name}-config-{timestamp}"
    
    # Create SageMaker model
    print(f"\n🤖 Creating SageMaker model: {model_name}")
    
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': f'763104351884.dkr.ecr.{aws_region}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311',
                'ModelDataUrl': model_data,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data,
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': aws_region
                }
            },
            ExecutionRoleArn=sagemaker_role
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return 1
    
    # Create endpoint configuration
    print(f"\n⚙️  Creating endpoint configuration: {endpoint_config_name}")
    
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 1.0
            }]
        )
        print("✅ Endpoint configuration created successfully")
    except Exception as e:
        print(f"❌ Error creating endpoint config: {e}")
        return 1
    
    # Create or update endpoint
    print(f"\n🚀 Deploying endpoint: {endpoint_name}")
    
    try:
        # Check if endpoint exists
        try:
            sagemaker.describe_endpoint(EndpointName=endpoint_name)
            print("⚠️  Endpoint exists, updating...")
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        except sagemaker.exceptions.ClientError:
            print("🆕 Creating new endpoint...")
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        
        print("\n" + "="*50)
        print("✅ DEPLOYMENT INITIATED")
        print("="*50)
        print(f"Endpoint name: {endpoint_name}")
        print(f"Instance type: ml.m5.large")
        print(f"Status: Creating (this may take 5-10 minutes)")
        print("\nMonitor status:")
        print(f"aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
        
    except Exception as e:
        print(f"❌ Error deploying endpoint: {e}")
        return 1
    
    print("\n✅ Deployment complete!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy model to SageMaker')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--endpoint', required=True, help='SageMaker endpoint name')
    parser.add_argument('--role', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    
    args = parser.parse_args()
    
    sys.exit(deploy_model(args.bucket, args.project, args.endpoint, args.role, args.region))
