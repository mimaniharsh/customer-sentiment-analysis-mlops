import os
import sys
import argparse
import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

def preprocess_and_upload(bucket_name, project_name, aws_region):
    """Download raw data from S3, preprocess, and upload preprocessed splits"""
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Download raw data with project path
    raw_key = f"{project_name}/data/raw/Tweets.csv"
    print(f"📥 Downloading raw data from s3://{bucket_name}/{raw_key}...")
    
    obj = s3_client.get_object(Bucket=bucket_name, Key=raw_key)
    df = pd.read_csv(obj['Body'])
    print(f"✅ Loaded {len(df)} rows")
    
    # Preprocessing
    print("🔄 Preprocessing data...")
    dfFinal = df[['airline_sentiment', 'text']].copy()
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    dfFinal['label'] = dfFinal['airline_sentiment'].map(sentiment_map)
    dfFinal.drop(columns=['airline_sentiment'], inplace=True)
    
    # Remove duplicates
    dfFinal = dfFinal.drop_duplicates().reset_index(drop=True)
    print(f"✅ After removing duplicates: {len(dfFinal)} rows")
    
    # Stratified split
    train_val, test = train_test_split(
        dfFinal, test_size=0.2, random_state=42, stratify=dfFinal['label']
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=42, stratify=train_val['label']
    )
    
    print(f"📊 Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Upload splits to S3 with project path
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        csv_buffer = StringIO()
        split_df.to_csv(csv_buffer, index=False)
        s3_key = f"{project_name}/data/processed/{split_name}.csv"
        print(f"📤 Uploading {s3_key}...")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
    
    print("✅ Preprocessing complete!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data and upload to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--project', required=True, help='Project name/prefix')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    
    args = parser.parse_args()
    
    sys.exit(preprocess_and_upload(args.bucket, args.project, args.region))
