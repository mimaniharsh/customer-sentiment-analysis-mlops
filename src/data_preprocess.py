import os
import sys
import boto3
import pandas as pd
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split

def preprocess_and_upload():
    """Download raw data from S3, preprocess, and upload preprocessed splits"""
    
    bucket_name = os.environ['S3_BUCKET']
    aws_region = os.environ['AWS_REGION']
    
    s3_client = boto3.client('s3', region_name=aws_region)
    
    print("Downloading raw data from S3...")
    obj = s3_client.get_object(Bucket=bucket_name, Key='data/raw/Tweets.csv')
    df = pd.read_csv(obj['Body'])
    print(f"Loaded {len(df)} rows")
    
    # Preprocessing
    print("Preprocessing data...")
    dfFinal = df[['airline_sentiment', 'text']].copy()
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    dfFinal['label'] = dfFinal['airline_sentiment'].map(sentiment_map)
    dfFinal.drop(columns=['airline_sentiment'], inplace=True)
    
    # Remove duplicates
    dfFinal = dfFinal.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {len(dfFinal)} rows")
    
    # Stratified split
    train_val, test = train_test_split(
        dfFinal, test_size=0.2, random_state=42, stratify=dfFinal['label']
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=42, stratify=train_val['label']
    )
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Upload splits to S3
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        csv_buffer = StringIO()
        split_df.to_csv(csv_buffer, index=False)
        s3_key = f"data/preprocessed/{split_name}.csv"
        print(f"Uploading {s3_key}...")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
    
    print("Preprocessing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(preprocess_and_upload())
