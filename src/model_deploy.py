import os
import sys
import json
import argparse
import boto3
import tarfile
import shutil
from datetime import datetime

def create_sagemaker_model(bucket_name, project_name, sagemaker_role, aws_region):
    """Create SageMaker model from trained artifacts"""
    
    print("="*70)
    print("📦 PREPARING SAGEMAKER MODEL ARTIFACT")
    print("="*70)
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download model files
    print("📥 Downloading model files from S3...")
    os.makedirs('./model', exist_ok=True)
    
    model_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 
                   'vocab.txt', 'special_tokens_map.json', 'tokenizer.json']
    
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
    
    # Create requirements for inference
    requirements = '''torch==2.3.1
transformers==4.44.2
'''
    with open('./model/requirements.txt', 'w') as f:
        f.write(requirements)
    print("   ✅ requirements.txt")
    
    # Create model.tar.gz
    print("\n🗜️  Creating model.tar.gz...")
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    # Upload to S3
    deploy_key = f"{project_name}/models/distilbert/model.tar.gz"
    print(f"📤 Uploading to s3://{bucket_name}/{deploy_key}")
    s3_client.upload_file('model.tar.gz', bucket_name, deploy_key)
    
    model_data_url = f"s3://{bucket_name}/{deploy_key}"
    
    # Create SageMaker model
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{project_name}-model-{timestamp}"
    
    print(f"\n🤖 Creating SageMaker model: {model_name}")
    
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
        print("✅ SageMaker model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        raise
    
    # Cleanup
    shutil.rmtree('./model')
    os.remove('model.tar.gz')
    
    return model_name, model_data_url

def create_endpoint_config(model_name, aws_region):
    """Create SageMaker endpoint configuration"""
    
    print("\n⚙️  Creating endpoint configuration...")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    config_name = f"sentiment-config-{timestamp}"
    
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
        print(f"✅ Endpoint config created: {config_name}")
    except Exception as e:
        print(f"❌ Error creating endpoint config: {e}")
        raise
    
    return config_name

def create_or_update_endpoint(config_name, endpoint_name, aws_region):
    """Create or update SageMaker endpoint"""
    
    print(f"\n🚀 Deploying endpoint: {endpoint_name}")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    try:
        # Check if endpoint exists
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print("⚠️  Endpoint exists, updating...")
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
    
    return endpoint_name

def wait_for_endpoint(endpoint_name, aws_region):
    """Wait for endpoint to be InService"""
    
    print(f"\n⏳ Waiting for endpoint {endpoint_name} to be InService...")
    
    sagemaker = boto3.client('sagemaker', region_name=aws_region)
    
    import time
    max_wait_time = 600  # 10 minutes
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            print(f"   Status: {status} ({elapsed_time}s elapsed)")
            
            if status == 'InService':
                print("✅ Endpoint is ready!")
                return True
            elif status in ['Failed', 'RollingBack']:
                failure_reason = response.get('FailureReason', 'Unknown')
                raise Exception(f"Endpoint creation failed: {failure_reason}")
            
            time.sleep(30)
            elapsed_time += 30
        except Exception as e:
            if 'Could not find endpoint' in str(e):
                print(f"   Endpoint not yet created, waiting...")
                time.sleep(30)
                elapsed_time += 30
            else:
                raise
    
    raise Exception(f"Endpoint creation timed out after {max_wait_time}s")

def deploy_model(bucket_name, project_name, endpoint_name, sagemaker_role, aws_region, output_file):
    """Deploy model to SageMaker endpoint"""
    
    print("="*70)
    print("🚀 MODEL DEPLOYMENT")
    print("="*70)
    
    # Create model
    model_name, model_uri = create_sagemaker_model(
        bucket_name, project_name, sagemaker_role, aws_region
    )
    
    # Create endpoint config
    config_name = create_endpoint_config(model_name, aws_region)
    
    # Create/update endpoint
    endpoint_name = create_or_update_endpoint(config_name, endpoint_name, aws_region)
    
    # Wait for endpoint
    wait_for_endpoint(endpoint_name, aws_region)
    
    # Save deployment info
    deployment_info = {
        'model_name': model_name,
        'config_name': config_name,
        'endpoint_name': endpoint_name,
        'model_uri': model_uri,
        'timestamp': datetime.now().isoformat(),
        'region': aws_region
    }
    
    with open(output_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"📍 Endpoint: {endpoint_name}")
    print(f"🌍 Region: {aws_region}")
    print(f"💾 Deployment info saved to: {output_file}")
    print("="*70)
    
    return deployment_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy model to SageMaker')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--endpoint', required=True, help='SageMaker endpoint name')
    parser.add_argument('--role', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    parser.add_argument('--output-file', default='deployment_info.json', help='Output file for deployment info')
    
    args = parser.parse_args()
    
    try:
        deploy_model(
            args.bucket,
            args.project,
            args.endpoint,
            args.role,
            args.region,
            args.output_file
        )
        sys.exit(0)
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
