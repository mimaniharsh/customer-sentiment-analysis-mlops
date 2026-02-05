# Machine Learning Pipeline with GitHub Actions and SageMaker

Production-ready ML pipeline for sentiment analysis using DistilBERT with automated deployment to AWS SageMaker.

## Architecture

```
GitHub Actions → S3 (Data) → SageMaker Training → Model Evaluation → SageMaker Endpoint
```

## Workflow Sequence

1. **Data Download** (`01-data-download.yml`)
   - Downloads Twitter airline sentiment dataset from Kaggle
   - Uploads raw CSV to S3: `s3://term3-harsh/data/raw/`

2. **Data Preprocessing** (`02-data-preprocess.yml`)
   - Downloads raw data from S3
   - Cleans and splits data (70% train, 10% val, 20% test)
   - Uploads preprocessed splits to S3: `s3://term3-harsh/data/preprocessed/`

3. **Model Training** (`03-model-train.yml`)
   - Trains DistilBERT model with class balancing
   - 3 epochs with weighted loss function
   - Saves model artifacts to S3: `s3://term3-harsh/models/distilbert/`

4. **Model Evaluation** (`04-model-eval.yml`)
   - Evaluates model on test set
   - Generates classification report and confusion matrix
   - Saves metrics to S3

5. **Model Deployment** (`05-model-deploy.yml`)
   - Creates model.tar.gz with inference script
   - Deploys to SageMaker endpoint: `harsh-ml-crp-project-name-endpoint`
   - Instance type: ml.m5.large

## Setup

### GitHub Repository Variables

```
AWS_REGION: ap-south-1
S3_BUCKET: term3-harsh
PROJECT_NAME: harsh-ml-crp-project-name
ENDPOINT_NAME: harsh-ml-crp-project-name-endpoint
```

### GitHub Repository Secrets

```
AWS_ROLE_ARN: arn:aws:iam::251198450921:role/GitHubActionS3Role
SAGEMAKER_ROLE_ARN: arn:aws:iam::251198450921:role/SageMakerStudentExecutionRole
```

### IAM Roles Required

**GitHubActionS3Role** - Permissions:
- s3:PutObject
- s3:GetObject
- s3:ListBucket

**SageMakerStudentExecutionRole** - Permissions:
- sagemaker:*
- s3:GetObject
- s3:PutObject
- ecr:GetAuthorizationToken
- ecr:BatchGetImage

## Project Structure

```
.
├── .github/workflows/
│   ├── 01-data-download.yml
│   ├── 02-data-preprocess.yml
│   ├── 03-model-train.yml
│   ├── 04-model-eval.yml
│   └── 05-model-deploy.yml
├── src/
│   ├── data_download.py
│   ├── data_preprocess.py
│   ├── model_train.py
│   ├── model_eval.py
│   └── model_deploy.py
├── requirements.txt
└── README.md
```

## Running the Pipeline

### Automated (Sequential)
Push to main branch triggers the complete pipeline automatically.

### Manual Execution
1. Go to Actions tab
2. Select workflow (01-05)
3. Click "Run workflow"

Workflows run sequentially - each waits for previous completion.

## Model Details

- **Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Multi-class sentiment classification (negative, neutral, positive)
- **Max Length**: 128 tokens
- **Batch Size**: 16
- **Epochs**: 10
- **Class Balancing**: Weighted CrossEntropyLoss

## S3 Bucket Structure

```
s3://term3-harsh/
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

## Endpoint Usage

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')

response = runtime.invoke_endpoint(
    EndpointName='harsh-ml-crp-project-name-endpoint',
    ContentType='application/json',
    Body=json.dumps({'text': 'The flight was delayed by 3 hours!'})
)

result = json.loads(response['Body'].read())
print(result)
```

**Response Format**:
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

## Dependencies

- boto3==1.34.162
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.1
- torch==2.3.1
- transformers==4.44.2
- sagemaker==2.231.0
- kagglehub==0.3.0

## Monitoring

- **GitHub Actions**: Check workflow runs in Actions tab
- **S3**: Monitor data and model artifacts in S3 console
- **SageMaker**: Check endpoint status and metrics in SageMaker console

## Cost Optimization

- Endpoint runs on ml.m5.large (delete when not in use)
- Workflows use GitHub Actions free tier
- S3 storage costs minimal for project size

## Troubleshooting

**Workflow fails at data download**:
- Check AWS_ROLE_ARN has S3 permissions
- Verify S3_BUCKET exists

**Training fails**:
- Ensure SAGEMAKER_ROLE_ARN has required permissions
- Check S3 bucket has preprocessed data

**Deployment fails**:
- Verify endpoint name doesn't conflict
- Check SageMaker quotas for ml.m5.large instances

## Cleanup

```bash
# Delete SageMaker endpoint
aws sagemaker delete-endpoint --endpoint-name harsh-ml-crp-project-name-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name harsh-ml-crp-project-name-config-*

# Delete model
aws sagemaker delete-model --model-name harsh-ml-crp-project-name-model-*
```

## License

MIT
