#!/usr/bin/env python3
"""
Model Deployment Script
Deploys SageMaker-trained models to endpoints
"""
import os
import sys
import json
import argparse
import boto3
import tarfile
import shutil
from datetime import datetime

def find_latest_model(s3_client, bucket_name, project_name):
    """Find latest trained model in S3"""
    
    print("🔍 Searching for trained model...")
    prefix = f"{project_name}/models/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            model_files = [
                obj for obj in response['Contents'] 
                if obj['Key'].endswith('model.tar.gz') and '/output/' in obj['Key']
            ]
            
            if model_files:
                latest = max(model_files, key=lambda x: x['LastModified'])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found model: {model_uri}")
                print(f"   Last modified: {latest['LastModified']}")
                return model_uri
    except Exception as e:
        print(f"❌ Error: {e}")
    
    raise FileNotFoundError(
        f"No trained model found in s3://{bucket_name}/{project_name}/models/. "
        "Please run training workflow first."
    )

def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri):
    """Download, add inference script, and repackage model"""
    
    print("\n📦 Preparing model for deployment...")
    
    # Download model
    model_key = model_uri.replace(f"s3://{bucket_name}/", "")
    print(f"📥 Downloading {model_key}...")
    s3_client.download_file(bucket_name, model_key, 'model.tar.gz')
    
    # Extract
    print("📂 Extracting model...")
    os.makedirs('./model', exist_ok=True)
    with tarfile.open('model.tar.gz', 'r:gz') as tar:
        tar.extractall('./model')
    
    os.remove('model.tar.gz')
    
    # Add inference script
    print("📝 Creating inference script...")
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
    
    # Add requirements
    requirements = 'torch==2.3.1\ntransformers==4.44.2\n'
    with open('./model/requirements.txt', 'w') as f:
        f.write(requirements)
    print("   ✅ requirements.txt")
    
    # Repackage
    print("\n🗜️  Creating deployment artifact...")
    with tarfile.open('deploy_model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    # Upload
    deploy_key = f"{project_name}/models/deploy/model.tar.gz"
    print(f"📤 Uploading to s3://{bucket_name}/{deploy_key}")
    s3_client.upload_file('deploy_model.tar.gz', bucket_name, deploy_key)
    
    deploy_uri = f"s3://{bucket_name}/{deploy_key}"
    
    # Cleanup
    shutil.rmtree('./model')
    os.remove('deploy_model.tar.gz')
    
    return deploy_uri

def create_sagemaker_model(model_name, model_data_url, sagemaker_role, aws_region):
    """Create SageMaker model"""
    
    print(f"\n🤖 Creating SageMaker model: {model_name}")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    image_uri = f'763104351884.dkr.ecr.{aws_region}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311'
    
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': aws_region
                }
            },
            ExecutionRoleArn=sagemaker_role
        )
        print("✅ Model created")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def create_endpoint_config(config_name, model_name, aws_region):
    """Create endpoint configuration"""
    
    print(f"\n⚙️  Creating endpoint config: {config_name}")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': 'ml.m5.large',
                'InitialInstanceCount': 1,
                'InitialVariantWeight': 1.0
            }]
        )
        print("✅ Config created")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def create_or_update_endpoint(endpoint_name, config_name, aws_region):
    """Create or update endpoint"""
    
    print(f"\n🚀 Deploying endpoint: {endpoint_name}")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print("⚠️  Updating existing endpoint...")
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except sagemaker.exceptions.ClientError:
        print("🆕 Creating new endpoint...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

def wait_for_endpoint(endpoint_name, aws_region):
    """Wait for endpoint to be InService"""
    
    print(f"\n⏳ Waiting for endpoint...")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    import time
    max_wait = 600
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            print(f"   Status: {status} ({elapsed}s)")
            
            if status == 'InService':
                print("✅ Endpoint ready!")
                return True
            elif status in ['Failed', 'RollingBack']:
                reason = response.get('FailureReason', 'Unknown')
                raise Exception(f"Endpoint failed: {reason}")
            
            time.sleep(30)
            elapsed += 30
        except sagemaker.exceptions.ClientError:
            time.sleep(30)
            elapsed += 30
    
    raise Exception(f"Timeout after {max_wait}s")

def deploy_model(args):
    """Main deployment function"""
    
    print("="*70)
    print("🚀 MODEL DEPLOYMENT")
    print("="*70)
    
    s3_client = boto3.client('s3', region_name=args.region)
    
    # Find model
    model_uri = find_latest_model(s3_client, args.bucket, args.project)
    
    # Prepare for deployment
    deploy_uri = prepare_model_for_deployment(s3_client, args.bucket, args.project, model_uri)
    
    # Create resources
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{args.project}-model-{timestamp}"
    config_name = f"{args.project}-config-{timestamp}"
    
    create_sagemaker_model(model_name, deploy_uri, args.role, args.region)
    create_endpoint_config(config_name, model_name, args.region)
    create_or_update_endpoint(args.endpoint, config_name, args.region)
    wait_for_endpoint(args.endpoint, args.region)
    
    # Save deployment info
    deployment_info = {
        'model_name': model_name,
        'config_name': config_name,
        'endpoint_name': args.endpoint,
        'model_uri': deploy_uri,
        'source_model': model_uri,
        'timestamp': datetime.now().isoformat(),
        'region': args.region
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"📍 Endpoint: {args.endpoint}")
    print(f"🌍 Region: {args.region}")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--endpoint', required=True, help='Endpoint name')
    parser.add_argument('--role', required=True, help='SageMaker role ARN')
    parser.add_argument('--region', required=True, help='AWS region')
    parser.add_argument('--output-file', default='deployment_info.json', help='Output file')
    
    args = parser.parse_args()
    
    try:
        sys.exit(deploy_model(args))
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
