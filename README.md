# Airline Sentiment Analysis: End-to-End MLOps Pipeline

[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade MLOps pipeline for sentiment analysis on airline customer feedback, leveraging DistilBERT transformers with fully automated CI/CD deployment to AWS SageMaker.**

---

## 📊 Project Overview

This project implements a complete machine learning operations (MLOps) pipeline that analyzes sentiment in airline customer tweets from the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). The pipeline automatically trains, evaluates, and deploys a DistilBERT-based sentiment classifier using GitHub Actions and AWS SageMaker.

### Key Features

- 🤖 **Advanced NLP Model**: Fine-tuned DistilBERT transformer for 3-class sentiment classification
- 🔄 **Automated CI/CD**: Sequential GitHub Actions workflows for end-to-end automation
- ☁️ **Cloud-Native**: AWS S3 for data storage, SageMaker for training and deployment
- 📈 **Class Balancing**: Weighted loss function to handle imbalanced datasets
- 🚀 **Production-Ready**: RESTful API endpoint with JSON inference
- 📊 **Comprehensive Metrics**: Detailed evaluation reports with precision, recall, and F1-scores

### Business Impact

Airlines can leverage this system to:
- **Monitor customer satisfaction** in real-time from social media
- **Identify service issues** quickly through negative sentiment detection
- **Prioritize customer support** based on sentiment severity
- **Track brand reputation** across different airlines and time periods

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Kaggle Dataset │
│   (14,640 tweets)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                      │
├─────────────────────────────────────────────────────────────┤
│  01-data-download → 02-preprocess → 03-train → 04-eval → 05-deploy │
└────────┬─────────────────┬──────────────┬──────────────────┘
         │                 │              │
         ▼                 ▼              ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐
│   AWS S3 Bucket │  │  SageMaker  │  │ SageMaker Endpoint│
│  (Data Storage) │  │  (Training) │  │  (Inference API) │
└─────────────────┘  └─────────────┘  └──────────────────┘
```

---

## 📁 Dataset

**Source**: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

The dataset contains **14,640 tweets** about major US airlines, collected in February 2015. Each tweet is labeled with sentiment (positive, neutral, negative) and includes metadata about airline, timestamp, and specific complaint reasons.

### Dataset Characteristics

| Metric | Value |
|--------|-------|
| **Total Tweets** | 14,640 (14,485 after deduplication) |
| **Airlines Covered** | United, US Airways, American, Southwest, Delta, Virgin America |
| **Sentiment Distribution** | Negative: 62.7%, Neutral: 21.1%, Positive: 16.2% |
| **Time Period** | February 16-24, 2015 |
| **Features Used** | `text` (tweet content), `airline_sentiment` (label) |

### Data Split Strategy

- **Training Set**: 70% (10,139 samples)
- **Validation Set**: 10% (1,449 samples)  
- **Test Set**: 20% (2,897 samples)

*Stratified sampling ensures proportional class distribution across all splits.*

---

## 🔄 Pipeline Workflow

1. **Data Download** (`01-data-download.yml`)
   - Downloads Twitter airline sentiment dataset from Kaggle
   - Uploads raw CSV to S3: `s3://your-ml-project-bucket/data/raw/`

2. **Data Preprocessing** (`02-data-preprocess.yml`)
   - Downloads raw data from S3
   - Cleans and splits data (70% train, 10% val, 20% test)
   - Uploads preprocessed splits to S3: `s3://your-ml-project-bucket/data/preprocessed/`

3. **Model Training** (`03-model-train.yml`)
   - Trains DistilBERT model with class balancing
   - 3 epochs with weighted loss function
   - Saves model artifacts to S3: `s3://your-ml-project-bucket/models/distilbert/`

4. **Model Evaluation** (`04-model-eval.yml`)
   - Evaluates model on test set
   - Generates classification report and confusion matrix
   - Saves metrics to S3

5. **Model Deployment** (`05-model-deploy.yml`)
   - Creates model.tar.gz with inference script
   - Deploys to SageMaker endpoint: `airline-sentiment-endpoint`
   - Instance type: ml.m5.large

---

## ⚙️ Setup & Configuration

### Prerequisites

- AWS Account with IAM permissions
- GitHub repository with Actions enabled
- S3 bucket created (`your-ml-project-bucket`)
- Python 3.11+

### 1️⃣ Configure GitHub Repository Variables

Navigate to: `Settings → Secrets and variables → Actions → Variables`

| Variable Name | Example Value | Description |
|---------------|---------------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for resources |
| `S3_BUCKET` | `your-ml-project-bucket` | S3 bucket for data/models |
| `PROJECT_NAME` | `airline-sentiment-mlops` | Unique project identifier |
| `ENDPOINT_NAME` | `airline-sentiment-endpoint` | SageMaker endpoint name |

> 💡 Choose unique names to avoid conflicts with existing AWS resources.

### 2️⃣ Configure GitHub Repository Secrets

Navigate to: `Settings → Secrets and variables → Actions → Secrets`

| Secret Name | Example Value | Description |
|-------------|---------------|-------------|
| `AWS_ROLE_ARN` | `arn:aws:iam::<account-id>:role/GitHubActionS3Role` | IAM role for data operations |
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::<account-id>:role/SageMakerExecutionRole` | IAM role for ML operations |

> ⚠️ **Security Note**: Replace `<account-id>` with your AWS account ID. Never commit actual ARNs to version control.

### 3️⃣ AWS IAM Configuration

#### GitHubActionS3Role

**Purpose**: Allows GitHub Actions to read/write S3 data

**Trust Policy**:
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
        "token.actions.githubusercontent.com:sub": "repo:<github-username>/<repo-name>:*"
      }
    }
  }]
}
```

> 💡 Replace `<account-id>`, `<github-username>`, and `<repo-name>` with your actual values.

**Permissions**:
- `AmazonS3FullAccess` (or custom policy with s3:PutObject, s3:GetObject, s3:ListBucket)

#### SageMakerStudentExecutionRole

**Purpose**: Allows SageMaker to train models and create endpoints

**Permissions**:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AmazonEC2ContainerRegistryReadOnly`

### 4️⃣ Create GitHub OIDC Provider (if not exists)

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

---

## 🚀 Running the Pipeline

### Automated (Sequential)
Push to main branch triggers the complete pipeline automatically.

### Manual Execution
1. Go to Actions tab
2. Select workflow (01-05)
3. Click "Run workflow"

---

## 🤖 Model Architecture & Training

### Base Model
**DistilBERT** (`distilbert-base-uncased`) - A distilled version of BERT that retains 97% of language understanding while being 40% smaller and 60% faster.

### Model Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Architecture** | DistilBERT Transformer | Optimal balance of performance and efficiency |
| **Task** | 3-class classification | Negative, Neutral, Positive sentiment |
| **Max Sequence Length** | 128 tokens | Covers 95%+ of tweet lengths |
| **Batch Size** | 16 | Optimized for GitHub Actions runners |
| **Training Epochs** | 3 | Prevents overfitting on small dataset |
| **Optimizer** | AdamW | Weight decay: 0.01 |
| **Learning Rate** | 5e-5 (default) | Standard for BERT fine-tuning |
| **Warmup Steps** | 500 | Gradual learning rate increase |
| **Loss Function** | Weighted CrossEntropyLoss | Handles class imbalance |

### Class Balancing Strategy

To address the 62.7% negative class dominance, we employ **inverse frequency weighting**:

```python
Class Weights:
- Negative (0): 0.53
- Neutral (1):  1.56  
- Positive (2): 2.04
```

This ensures the model learns equally from all sentiment classes despite dataset imbalance.

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Weighted F1-Score**: Primary metric (accounts for class imbalance)
- **Precision & Recall**: Per-class performance
- **Confusion Matrix**: Detailed error analysis

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness across all classes
- **Weighted F1-Score**: Primary metric (accounts for class imbalance)
- **Precision & Recall**: Per-class performance analysis
- **Confusion Matrix**: Detailed error distribution

*Best model checkpoint is selected based on validation F1-score.*

### Expected Performance

Based on similar DistilBERT implementations on this dataset:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Accuracy** | 78-82% | Baseline performance |
| **F1-Score (Weighted)** | 76-80% | Accounts for imbalance |
| **Negative Class** | Precision: 82-85% | Majority class |
| **Neutral Class** | Precision: 68-72% | Challenging middle ground |
| **Positive Class** | Precision: 75-80% | Minority class |

> 📊 Actual results stored in `s3://your-ml-project-bucket/models/distilbert/metrics.json`

---

## 📦 S3 Data Storage Structure

```
s3://your-ml-project-bucket/
├── data/
│   ├── raw/
│   │   └── Tweets.csv
│   └── preprocessed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── models/
    └── distilbert/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer_config.json
        ├── vocab.txt
        ├── model.tar.gz
        └── metrics.json
```

---

## 🔌 API Usage

Once deployed, the SageMaker endpoint provides a RESTful API for real-time sentiment inference.

### Python Client Example

```python
import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Sample tweet for inference
tweet = "The flight was delayed by 3 hours and the staff was unhelpful!"

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='airline-sentiment-endpoint',  # Your endpoint name
    ContentType='application/json',
    Body=json.dumps({'text': tweet})
)

# Parse response
result = json.loads(response['Body'].read())
print(json.dumps(result, indent=2))
```

### Response Format

```json
{
  "prediction": "negative",
  "confidence": 0.9234,
  "probabilities": {
    "negative": 0.9234,
    "neutral": 0.0521,
    "positive": 0.0245
  }
}
```

### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Predicted sentiment class (`negative`, `neutral`, `positive`) |
| `confidence` | float | Confidence score for the predicted class (0-1) |
| `probabilities` | object | Probability distribution across all classes |

### Batch Inference

For processing multiple tweets:

```python
tweets = [
    "Great service, loved the flight!",
    "The baggage was lost, very disappointed",
    "Flight was on time, nothing special"
]

results = []
for tweet in tweets:
    response = runtime.invoke_endpoint(
        EndpointName='airline-sentiment-endpoint',
        ContentType='application/json',
        Body=json.dumps({'text': tweet})
    )
    results.append(json.loads(response['Body'].read()))

# Analyze results
for tweet, result in zip(tweets, results):
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {result['prediction']} ({result['confidence']:.2%})")
    print()
```

---

## 📂 Project Structure

```
.
├── .github/
│   └── workflows/           # CI/CD pipeline definitions
│       ├── 01-data-download.yml
│       ├── 02-data-preprocess.yml
│       ├── 03-model-train.yml
│       ├── 04-model-eval.yml
│       └── 05-model-deploy.yml
├── src/                     # Python source code
│   ├── data_download.py     # Kaggle → S3 download
│   ├── data_preprocess.py   # Data cleaning & splitting
│   ├── model_train.py       # DistilBERT training
│   ├── model_eval.py        # Model evaluation
│   └── model_deploy.py      # SageMaker deployment
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🔧 Technology Stack

### Machine Learning
- **Framework**: PyTorch 2.3.1
- **Model**: Hugging Face Transformers (DistilBERT)
- **Preprocessing**: scikit-learn, pandas, numpy

### Cloud Infrastructure
- **Compute**: AWS SageMaker (ml.m5.large)
- **Storage**: Amazon S3
- **IAM**: AWS Identity and Access Management

### DevOps & CI/CD
- **Pipeline**: GitHub Actions
- **Authentication**: OpenID Connect (OIDC)
- **Version Control**: Git

### Languages & Tools
- **Primary**: Python 3.11
- **SDK**: Boto3 (AWS), SageMaker SDK
- **Package Management**: pip

---

## 📊 Monitoring & Observability

### GitHub Actions Dashboard

Monitor workflow execution in real-time:
1. Navigate to `Actions` tab
2. View workflow runs, logs, and artifacts
3. Check for errors or warnings

### AWS CloudWatch Metrics

Track SageMaker endpoint performance:
- **Invocations**: Request count
- **ModelLatency**: Inference time (ms)
- **Overhead**: Pre/post-processing time
- **Errors**: Failed requests

### S3 Data Audit

```bash
# View raw data
aws s3 ls s3://your-ml-project-bucket/data/raw/

# View preprocessed splits
aws s3 ls s3://your-ml-project-bucket/data/preprocessed/

# View model artifacts
aws s3 ls s3://your-ml-project-bucket/models/distilbert/
```

### SageMaker Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name airline-sentiment-endpoint \
  --query 'EndpointStatus'
```

---

## 💰 Cost Analysis

### GitHub Actions
- **Free Tier**: 2,000 minutes/month
- **This Project**: ~30 minutes per full pipeline run
- **Estimated Cost**: $0/month (within free tier)

### AWS S3
- **Storage**: ~50 MB (raw + preprocessed data + model)
- **Requests**: ~100 per pipeline run
- **Estimated Cost**: ~$0.10/month

### AWS SageMaker
- **Training**: ml.m5.large (~30 minutes)
  - **Cost**: ~$0.10 per run
- **Endpoint**: ml.m5.large (24/7)
  - **Cost**: ~$72/month if always running
  - **💡 Recommendation**: Delete endpoint when not in use

**Monthly Total**: ~$0.20 (excluding endpoint) or ~$72 (with 24/7 endpoint)

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### ❌ Error: "Input required and not supplied: aws-region"

**Cause**: GitHub Actions workflow missing AWS region configuration

**Solution**:
```bash
# Ensure workflow files have hardcoded region:
aws-region: us-east-1  # Not ${{ vars.AWS_REGION }}
```

#### ❌ Error: "AccessDenied" or "AssumeRole failed"

**Cause**: IAM role not configured for GitHub OIDC

**Solution**:
1. Verify OIDC provider exists in AWS IAM
2. Check trust relationship includes your repository
3. Update `YOUR_USERNAME/YOUR_REPO` in trust policy

```bash
# Check OIDC provider
aws iam list-open-id-connect-providers
```

#### ❌ Workflow: "Kaggle API not authenticated"

**Cause**: `kagglehub` requires authentication

**Solution**: The library auto-handles this. If error persists, add Kaggle credentials to GitHub Secrets:
```
KAGGLE_USERNAME: your_username
KAGGLE_KEY: your_api_key
```

#### ❌ Training: "CUDA out of memory"

**Cause**: GitHub Actions runners don't have GPU

**Solution**: Already configured for CPU (`torch.device('cpu')`). No action needed.

#### ❌ Deployment: "Endpoint already exists"

**Cause**: Previous endpoint with same name still active

**Solution**:
```bash
aws sagemaker delete-endpoint --endpoint-name airline-sentiment-endpoint
```

#### ❌ S3: "NoSuchBucket"

**Cause**: S3 bucket doesn't exist

**Solution**:
```bash
aws s3 mb s3://your-ml-project-bucket --region us-east-1
```

---

## 🧹 Cleanup & Resource Management

### Delete SageMaker Endpoint (Stop Billing)

```bash
# Delete endpoint (stops inference)
aws sagemaker delete-endpoint \
  --endpoint-name airline-sentiment-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config \
  --endpoint-config-name airline-sentiment-config-<timestamp>

# Delete model
aws sagemaker delete-model \
  --model-name airline-sentiment-model-<timestamp>
```

### Clean S3 Bucket (Optional)

```bash
# Remove all data (careful!)
aws s3 rm s3://your-ml-project-bucket/ --recursive

# Remove specific folders
aws s3 rm s3://your-ml-project-bucket/data/ --recursive
aws s3 rm s3://your-ml-project-bucket/models/ --recursive
```

### Delete IAM Roles (When Done)

```bash
# Detach policies first
aws iam detach-role-policy \
  --role-name GitHubActionS3Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Delete role
aws iam delete-role --role-name GitHubActionS3Role
```

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `boto3` | 1.34.162 | AWS SDK for Python |
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `scikit-learn` | 1.5.1 | ML utilities & metrics |
| `torch` | 2.3.1 | Deep learning framework |
| `transformers` | 4.44.2 | Hugging Face transformers |
| `sagemaker` | 2.231.0 | SageMaker SDK |
| `kagglehub` | 0.3.0 | Kaggle dataset downloader |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Machine Learning Engineer | MLOps Specialist**

Experienced in building production ML pipelines with focus on:
- End-to-end model deployment automation
- Cloud-native architectures (AWS, GCP, Azure)
- NLP and transformer-based models
- CI/CD for ML systems

---

## 🙏 Acknowledgments

- **Dataset**: [Crowdflower Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Model**: [Hugging Face DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Infrastructure**: AWS SageMaker, GitHub Actions
- **Community**: Special thanks to the open-source ML community

---

## 📞 Support & Contact

### Getting Help

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/yourrepo/issues) with detailed reproduction steps
- 💡 **Feature Requests**: [Start a discussion](https://github.com/yourusername/yourrepo/discussions) to propose enhancements
- 📖 **Documentation**: Check the [Wiki](https://github.com/yourusername/yourrepo/wiki) for additional guides
- 💬 **Community**: Join discussions for Q&A and best practices

### Response Time

- Critical bugs: 24-48 hours
- Feature requests: 1-2 weeks
- General questions: 2-5 days

---

## 🗺️ Roadmap

### Phase 1: Foundation (Completed ✅)
- [x] End-to-end MLOps pipeline with GitHub Actions
- [x] DistilBERT fine-tuning with class balancing
- [x] SageMaker deployment automation
- [x] REST API endpoint for inference

### Phase 2: Model Improvements (Q1 2025)
- [ ] Experiment with BERT-Large and RoBERTa architectures
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Add ensemble methods for improved accuracy
- [ ] Multi-task learning for sentiment + reason classification

### Phase 3: Production Enhancements (Q2 2025)
- [ ] Real-time monitoring dashboard (CloudWatch + Grafana)
- [ ] A/B testing framework for model comparison
- [ ] Automated model retraining on data drift detection
- [ ] Model versioning and rollback with MLflow

### Phase 4: Scaling & Advanced Features (Q3 2025)
- [ ] Multi-language sentiment analysis support
- [ ] Streaming inference with Kinesis Data Streams
- [ ] Docker containerization for portability
- [ ] Kubernetes deployment option (EKS)
- [ ] Cost optimization with SageMaker Serverless Inference

### Phase 5: Enterprise Features (Q4 2025)
- [ ] Integration with social media APIs (Twitter, Facebook)
- [ ] Custom training on client-specific datasets
- [ ] White-label API solution
- [ ] Compliance certifications (SOC 2, HIPAA)

---

**⭐ If you find this project helpful, please consider giving it a star!**
