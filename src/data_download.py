import os
import sys
import argparse
import boto3
import kagglehub
from pathlib import Path

def download_and_upload_data(bucket_name, project_name, aws_region):
    """Download dataset from Kaggle and upload to S3"""
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=aws_region)
    
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    print(f"Dataset downloaded to: {path}")
    
    # Upload to S3 with project path
    csv_file = Path(path) / "Tweets.csv"
    s3_key = f"{project_name}/data/raw/Tweets.csv"
    
    print(f"Uploading {csv_file} to s3://{bucket_name}/{s3_key}")
    s3_client.upload_file(str(csv_file), bucket_name, s3_key)
    print("✅ Upload complete!")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset and upload to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    sys.exit(download_and_upload_data(args.bucket, args.project, args.region))
