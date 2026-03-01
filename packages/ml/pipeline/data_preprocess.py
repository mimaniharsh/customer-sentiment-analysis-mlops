#!/usr/bin/env python3
"""
packages/ml/pipeline/data_preprocess.py

Upgraded preprocessing pipeline:
  - Real text cleaning: URLs, @mentions, HTML entities, repeated chars
  - NaN / empty text guard
  - Minimum text length filter (>= 5 chars)
  - Corrected stratified split (70 / 10 / 20)
  - Label distribution logging
  - Supports airline tweet schema AND generic (text, label) CSV
"""
import re
import sys
import html
import argparse
import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                             # &amp; -> &
    text = re.sub(r"https?://\S+|www\.\S+", "", text)     # strip URLs
    text = re.sub(r"@\w+", "", text)                       # strip @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                  # #word -> word
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)             # loooove -> love
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_upload(bucket_name: str, project_name: str, aws_region: str) -> int:
    s3_client = boto3.client("s3", region_name=aws_region)

    raw_key = f"{project_name}/data/raw/Tweets.csv"
    print(f"📥 Downloading raw data from s3://{bucket_name}/{raw_key}...")
    obj = s3_client.get_object(Bucket=bucket_name, Key=raw_key)
    df = pd.read_csv(obj["Body"])
    print(f"✅ Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    # Detect schema
    if "airline_sentiment" in df.columns and "text" in df.columns:
        df = df[["airline_sentiment", "text"]].copy()
        df["label"] = df["airline_sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
        df.drop(columns=["airline_sentiment"], inplace=True)
    elif "text" in df.columns and "label" in df.columns:
        df = df[["text", "label"]].copy()
    else:
        raise ValueError(f"Unrecognised schema. Expected ('airline_sentiment','text') or ('text','label'). Found: {list(df.columns)}")

    # Clean text
    print(f"\n🔄 Cleaning {len(df):,} texts...")
    df["text"] = df["text"].apply(clean_text)

    # Quality guards
    before = len(df)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df[df["label"].isin([0, 1, 2])]
    df = df[df["text"].str.len() >= 5]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"   Dropped {before - len(df):,} rows. Remaining: {len(df):,}")

    # Label distribution
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    print("\n📊 Label distribution:")
    for lbl, count in df["label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"   {label_names.get(int(lbl)):10s}: {count:6,}  ({pct:.1f}%)")

    # Stratified 70 / 10 / 20 split
    # Step 1: 20% test
    train_val, test = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    # Step 2: 12.5% of remainder = 10% of total for val
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val["label"])

    print(f"\n✂️  Splits — Train: {len(train):,} ({len(train)/len(df)*100:.0f}%)  Val: {len(val):,} ({len(val)/len(df)*100:.0f}%)  Test: {len(test):,} ({len(test)/len(df)*100:.0f}%)")

    # Upload
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        buf = StringIO()
        split_df.to_csv(buf, index=False)
        s3_key = f"{project_name}/data/processed/{split_name}.csv"
        print(f"📤 Uploading {s3_key} ({len(split_df):,} rows)...")
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=buf.getvalue())

    print("\n✅ Preprocessing complete!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()
    sys.exit(preprocess_and_upload(args.bucket, args.project, args.region))
