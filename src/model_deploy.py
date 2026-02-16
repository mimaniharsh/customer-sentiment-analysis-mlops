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
import time

def find_latest_model(s3_client, bucket_name, project_name):
    print("🔍 Searching for trained model...")
    prefix = f"{project_name}/models/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            model_files = [obj for obj in response['Contents'] if obj['Key'].endswith('model.tar.gz') and '/output/' in obj['Key']]
            if model_files:
                latest = max(model_files, key=lambda x: x['LastModified'])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found model: {model_uri}")
                return model_uri
    except Exception as e:
        print(f"❌ Error: {e}")
    raise FileNotFoundError("No trained model found.")

def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri):
    print("\n📦 Preparing model for deployment...")
    model_key = model_uri.replace(f"s3://{bucket_name}/", "")
    s3_client.download_file(bucket_name, model_key, 'model.tar.gz')
    
    os.makedirs('./model', exist_ok=True)
    with tarfile.open('model.tar.gz', 'r:gz') as tar:
        tar.extractall('./model')
    os.remove('model.tar.gz')
    
    os.makedirs('./model/code', exist_ok=True)
    
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
    
    # FIXED: Restored correct dictionary iteration here!
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
    with open('./model/code/inference.py', 'w') as f:
        f.write(inference_code)
    
    print("📝 Creating requirements.txt...")
    requirements = 'transformers==4.44.2\n'
    with open('./model/code/requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("\n🗜️  Creating deployment artifact...")
    with tarfile.open('deploy_model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    deploy_key = f"{project_name}/models/deploy/model.tar.gz"
    s3_client.upload_file('deploy_model.tar.gz', bucket_name, deploy_key)
    deploy_uri = f"s3://{bucket_name}/{deploy_key}"
    
    shutil.rmtree('./model')
    os.remove('deploy_model.tar.gz')
    return deploy_uri

def create_sagemaker_model(model_name, model_data_url, sagemaker_role, aws_region):
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
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': aws_region,
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '600',
                    'TS_DEFAULT_RESPONSE_TIMEOUT': '600'
                }
            },
            ExecutionRoleArn=sagemaker_role
        )
        print("✅ Model created")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def create_endpoint_config(config_name, model_name, aws_region):
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
    print(f"\n🚀 Deploying endpoint: {endpoint_name}")
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        if status == 'Failed':
            print("⚠️  Endpoint is in Failed state. Deleting and recreating...")
            sagemaker.delete_endpoint(EndpointName=endpoint_name)
            time.sleep(10)
            sagemaker.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        else:
            print("⚠️  Updating existing endpoint...")
            sagemaker.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except sagemaker.exceptions.ClientError:
        print("🆕 Creating new endpoint...")
        sagemaker.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

def wait_for_endpoint(endpoint_name, aws_region):
    print(f"\n⏳ Waiting for endpoint...")
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
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
    print("="*70)
    print("🚀 MODEL DEPLOYMENT")
    print("="*70)
    s3_client = boto3.client('s3', region_name=args.region)
    model_uri = find_latest_model(s3_client, args.bucket, args.project)
    deploy_uri = prepare_model_for_deployment(s3_client, args.bucket, args.project, model_uri)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{args.project}-model-{timestamp}"
    config_name = f"{args.project}-config-{timestamp}"
    
    create_sagemaker_model(model_name, deploy_uri, args.role, args.region)
    create_endpoint_config(config_name, model_name, args.region)
    create_or_update_endpoint(args.endpoint, config_name, args.region)
    wait_for_endpoint(args.endpoint, args.region)
    
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
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--output-file', default='deployment_info.json')
    args = parser.parse_args()
    try:
        sys.exit(deploy_model(args))
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        sys.exit(1)