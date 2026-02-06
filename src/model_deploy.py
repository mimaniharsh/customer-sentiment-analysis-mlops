import os
import sys
import json
import argparse
import boto3
import tarfile
import shutil
from datetime import datetime

def find_latest_model_in_s3(s3_client, bucket_name, project_name):
    """Find the latest trained model (supports SageMaker training jobs and direct uploads)"""
    
    print("🔍 Searching for trained model...")
    
    # Check SageMaker training job outputs
    sagemaker_prefix = f"{project_name}/models/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=sagemaker_prefix
        )
        
        if 'Contents' in response:
            # Find model.tar.gz files from training jobs
            model_files = [
                obj for obj in response['Contents'] 
                if obj['Key'].endswith('model.tar.gz') and '/output/' in obj['Key']
            ]
            
            if model_files:
                # Get the most recent one
                latest = max(model_files, key=lambda x: x['LastModified'])
                model_uri = f"s3://{bucket_name}/{latest['Key']}"
                print(f"✅ Found SageMaker training output: {model_uri}")
                return model_uri, 'sagemaker'
    except Exception as e:
        print(f"⚠️  Error searching SageMaker outputs: {e}")
    
    # Check direct upload location
    direct_prefix = f"{project_name}/models/distilbert/"
    print(f"🔍 Checking direct upload: s3://{bucket_name}/{direct_prefix}")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=direct_prefix
        )
        
        if 'Contents' in response and len(response['Contents']) > 1:
            print(f"✅ Found model files in direct upload location")
            return f"s3://{bucket_name}/{direct_prefix}", 'direct'
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    raise FileNotFoundError(
        f"No trained model found in s3://{bucket_name}/{project_name}/models/. "
        "Please run training workflow first."
    )

def prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri, source_type):
    """Download and prepare model for SageMaker deployment"""
    
    print("\n📦 Preparing model for deployment...")
    
    # Create temp directory
    os.makedirs('./model', exist_ok=True)
    
    if source_type == 'sagemaker':
        # Download existing model.tar.gz from training job
        print(f"📥 Downloading from: {model_uri}")
        model_key = model_uri.replace(f"s3://{bucket_name}/", "")
        s3_client.download_file(bucket_name, model_key, 'model.tar.gz')
        
        # Extract it
        print("📂 Extracting model...")
        with tarfile.open('model.tar.gz', 'r:gz') as tar:
            tar.extractall('./model')
        
        os.remove('model.tar.gz')
        
    else:
        # Download individual files
        print(f"📥 Downloading model files...")
        prefix = model_uri.replace(f"s3://{bucket_name}/", "")
        
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            filename = key.replace(prefix, '')
            if filename:
                local_path = f'./model/{filename}'
                s3_client.download_file(bucket_name, key, local_path)
                print(f"   ✅ {filename}")
    
    # Add inference script
    print("\n📝 Creating inference script...")
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
    
    # Create requirements
    requirements = '''torch==2.3.1
transformers==4.44.2
'''
    with open('./model/requirements.txt', 'w') as f:
        f.write(requirements)
    print("   ✅ requirements.txt")
    
    # Create new model.tar.gz
    print("\n🗜️  Creating deployment artifact...")
    with tarfile.open('deploy_model.tar.gz', 'w:gz') as tar:
        tar.add('./model', arcname='.')
    
    # Upload to deployment location
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
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    return model_name

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
        print("✅ Config created successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    return config_name

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
    
    return endpoint_name

def wait_for_endpoint(endpoint_name, aws_region):
    """Wait for endpoint to be InService"""
    
    print(f"\n⏳ Waiting for endpoint to be ready...")
    
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
                print("✅ Endpoint is ready!")
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

def deploy_model(bucket_name, project_name, endpoint_name, sagemaker_role, aws_region, output_file):
    """Main deployment function"""
    
    print("="*70)
    print("🚀 MODEL DEPLOYMENT")
    print("="*70)
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Find the model
    model_uri, source_type = find_latest_model_in_s3(s3_client, bucket_name, project_name)
    
    # Prepare for deployment
    deploy_uri = prepare_model_for_deployment(s3_client, bucket_name, project_name, model_uri, source_type)
    
    # Create resources
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{project_name}-model-{timestamp}"
    config_name = f"{project_name}-config-{timestamp}"
    
    create_sagemaker_model(model_name, deploy_uri, sagemaker_role, aws_region)
    create_endpoint_config(config_name, model_name, aws_region)
    create_or_update_endpoint(endpoint_name, config_name, aws_region)
    wait_for_endpoint(endpoint_name, aws_region)
    
    # Save info
    deployment_info = {
        'model_name': model_name,
        'config_name': config_name,
        'endpoint_name': endpoint_name,
        'model_uri': deploy_uri,
        'source_model': model_uri,
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
    print("="*70)
    
    return deployment_info

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
        print(f"\n❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
