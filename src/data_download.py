import os
import sys
import boto3
import kagglehub
from pathlib import Path

def download_and_upload_data():
    """Download dataset from Kaggle and upload to S3"""
    
    # Environment variables
    bucket_name = os.environ['S3_BUCKET']
    aws_region = os.environ['AWS_REGION']
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=aws_region)
    
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    print(f"Dataset downloaded to: {path}")
    
    # Upload to S3
    csv_file = Path(path) / "Tweets.csv"
    s3_key = "data/raw/Tweets.csv"
    
    print(f"Uploading {csv_file} to s3://{bucket_name}/{s3_key}")
    s3_client.upload_file(str(csv_file), bucket_name, s3_key)
    print("Upload complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(download_and_upload_data())
