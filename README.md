# Customer Sentiment Analysis MLOps Pipeline

[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

> Production-grade MLOps pipeline for airline customer sentiment analysis using DistilBERT transformers with automated CI/CD deployment to AWS SageMaker.

## 📊 Project Overview

Automated sentiment analysis system analyzing airline customer feedback from Twitter. The pipeline handles data ingestion, preprocessing, model training, evaluation, and deployment with full CI/CD automation.

### Key Features

- 🤖 **DistilBERT Transformer**: Fine-tuned for 3-class sentiment classification
- 🔄 **Automated CI/CD**: Sequential GitHub Actions workflows
- ☁️ **AWS Integration**: S3 storage + SageMaker training & deployment
- ⚖️ **Class Balancing**: Weighted loss for imbalanced datasets
- 🚀 **REST API**: Production-ready inference endpoint
- 📊 **Comprehensive Metrics**: Precision, recall, F1-score evaluation

## 🏗️ Architecture

```
┌──────────────┐
│   Kaggle     │
│  Dataset     │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│           GitHub Actions CI/CD Pipeline                  │
├─────────────────────────────────────────────────────────┤
│  01-download → 02-preprocess → 03-train → 04-eval → 05-deploy │
└────┬──────────────┬──────────────┬────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌──────────┐   ┌─────────────┐
│ AWS S3  │   │SageMaker │   │  Endpoint   │
│ Storage │   │ Training │   │  (ml.m5)    │
└─────────┘   └──────────┘   └─────────────┘
```

## 📁 Dataset

**Source**: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

- **Size**: 14,640 tweets (14,485 after deduplication)
- **Classes**: Negative (62.7%), Neutral (21.1%), Positive (16.2%)
- **Split**: Train (70%), Val (10%), Test (20%)

## 🔄 Pipeline Workflow

### 01 - Data Download
- Downloads dataset from Kaggle
- Uploads to S3: `customer-sentiment-analysis-mlops/data/raw/`

### 02 - Data Preprocessing
- Cleans and tokenizes text
- Stratified train/val/test split
- Uploads to S3: `customer-sentiment-analysis-mlops/data/processed/`

### 03 - Model Training
- Fine-tunes DistilBERT (CPU optimized)
- 3 epochs with weighted loss
- Uploads model to S3: `customer-sentiment-analysis-mlops/models/distilbert/`

### 04 - Model Evaluation
- Tests on held-out test set
- Generates metrics and confusion matrix
- Saves results to S3

### 05 - Model Deployment
- Packages model with inference script
- Deploys to SageMaker endpoint (ml.m5.large)
- Creates REST API for real-time inference

## ⚙️ Setup

### Prerequisites

- AWS Account with IAM permissions
- GitHub repository with Actions enabled
- S3 bucket created
- Python 3.11+

### GitHub Variables

Navigate to: `Settings → Secrets and variables → Actions → Variables`

| Variable | Value |
|----------|-------|
| `S3_BUCKET` | Your S3 bucket name |
| `ENDPOINT_NAME` | Your SageMaker endpoint name |

### GitHub Secrets

Navigate to: `Settings → Secrets and variables → Actions → Secrets`

| Secret | Value |
|--------|-------|
| `AWS_ROLE_ARN` | `arn:aws:iam::<account-id>:role/GitHubActionsRole` |
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::<account-id>:role/SageMakerRole` |

### AWS IAM Setup

**Create OIDC Provider** (one-time):
```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**GitHubActionsRole Trust Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:<username>/<repo>:*"
      }
    }
  }]
}
```

**Permissions**:
- GitHubActionsRole: `AmazonS3FullAccess`
- SageMakerRole: `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`

## 🚀 Running the Pipeline

### Automated Execution
Push to `main` branch triggers the full pipeline automatically.

### Manual Execution
1. Go to **Actions** tab
2. Select workflow (01-05)
3. Click **Run workflow**

Workflows run sequentially - each waits for the previous to complete.

## 🔌 API Usage

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps({'text': 'Flight was delayed 3 hours!'})
)

result = json.loads(response['Body'].read())
print(result)
```

**Response**:
```json
{
  "prediction": "negative",
  "confidence": 0.92,
  "probabilities": {
    "negative": 0.92,
    "neutral": 0.05,
    "positive": 0.03
  }
}
```

## 📂 Project Structure

```
.
├── .github/workflows/     # CI/CD workflows
│   ├── 01-data-download.yml
│   ├── 02-data-preprocess.yml
│   ├── 03-model-train.yml
│   ├── 04-model-eval.yml
│   └── 05-model-deploy.yml
├── src/                   # Python source code
│   ├── data_download.py
│   ├── data_preprocess.py
│   ├── model_train.py
│   ├── model_eval.py
│   └── model_deploy.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 📊 S3 Structure

```
s3://your-bucket/
└── customer-sentiment-analysis-mlops/
    ├── data/
    │   ├── raw/Tweets.csv
    │   └── processed/{train,val,test}.csv
    └── models/
        └── distilbert/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer files
            ├── model.tar.gz
            └── metrics.json
```

## 💰 Cost Analysis

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| GitHub Actions | 2000 min free | $0 |
| S3 Storage | ~50 MB | ~$0.10 |
| SageMaker Training | ~30 min | ~$0.10/run |
| SageMaker Endpoint | ml.m5.large 24/7 | ~$72 |

**💡 Recommendation**: Delete endpoint when not in use to avoid costs.

## 🛠️ Troubleshooting

### Error: "aws-region not supplied"
Workflows have hardcoded `us-east-1`. Verify the region matches your resources.

### Error: "AssumeRole failed"
1. Verify OIDC provider exists
2. Check trust policy includes your repo
3. Confirm secrets are set correctly

### Error: "NoSuchBucket"
```bash
aws s3 mb s3://your-bucket --region us-east-1
```

### Training takes too long
Expected: 20-30 minutes on GitHub Actions CPU runners. This is normal for transformer models.

## 🧹 Cleanup

```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name

# Delete endpoint config
aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>

# Delete model
aws sagemaker delete-model --model-name <model-name>

# Empty S3 bucket
aws s3 rm s3://your-bucket/customer-sentiment-analysis-mlops/ --recursive
```

## 📚 Tech Stack

- **ML Framework**: PyTorch 2.3.1, Hugging Face Transformers 4.44.2
- **Cloud**: AWS S3, SageMaker
- **CI/CD**: GitHub Actions
- **Language**: Python 3.11

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push and open a Pull Request

## 📄 License

MIT License

## 👨‍💻 Author

Machine Learning Engineer | MLOps Specialist

Experienced in building production ML pipelines with focus on:
- End-to-end model deployment automation
- Cloud-native architectures
- NLP and transformer models
- CI/CD for ML systems

---

**⭐ Star this repo if you find it helpful!**
