# 🚀 Codespaces Deployment Instructions

## Quick Setup (5 minutes)

### Step 1: Upload to Codespaces

1. **Create GitHub Repository**
   - Go to github.com → New Repository
   - Name: `customer-sentiment-analysis-mlops`
   - Public or Private
   - Don't initialize with README

2. **Open Codespaces**
   - Click `Code` → `Codespaces` → `Create codespace on main`

3. **Upload Files**
   In Codespaces terminal:
   ```bash
   # Upload the downloaded 'final-project' folder to Codespaces
   # Then move into it:
   cd final-project
   
   # Initialize git
   git init
   git add .
   git commit -m "Initial MLOps pipeline setup"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/customer-sentiment-analysis-mlops.git
   git push -u origin main
   ```

### Step 2: Configure GitHub Secrets

**Variables** (`Settings → Secrets and variables → Actions → Variables`):
```
Name: S3_BUCKET
Value: your-bucket-name

Name: ENDPOINT_NAME  
Value: customer-sentiment-endpoint
```

**Secrets** (`Settings → Secrets and variables → Actions → Secrets`):
```
Name: AWS_ROLE_ARN
Value: arn:aws:iam::YOUR_ACCOUNT_ID:role/GitHubActionsRole

Name: SAGEMAKER_ROLE_ARN
Value: arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
```

### Step 3: AWS Setup

**Create S3 Bucket**:
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

**Create OIDC Provider** (one-time):
```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**Create GitHubActionsRole**:
1. IAM → Roles → Create Role
2. Custom trust policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:YOUR_USERNAME/customer-sentiment-analysis-mlops:*"
      }
    }
  }]
}
```
3. Attach policy: `AmazonS3FullAccess`
4. Name: `GitHubActionsRole`

**Create SageMakerExecutionRole**:
1. IAM → Roles → Create Role
2. Trust entity: SageMaker
3. Attach policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
4. Name: `SageMakerExecutionRole`

### Step 4: Run Pipeline

1. Go to **Actions** tab
2. Click **01 - Data Download**
3. Click **Run workflow** → **Run workflow**
4. Wait for completion (~2 min)
5. Workflows 02-05 will run automatically in sequence

## 📊 Expected Results

**Total Pipeline Time**: ~30-40 minutes

| Workflow | Duration | Output |
|----------|----------|--------|
| 01 - Data Download | 2-3 min | Raw data in S3 |
| 02 - Preprocessing | 1-2 min | Train/val/test splits |
| 03 - Model Training | 20-30 min | Trained DistilBERT model |
| 04 - Evaluation | 2-3 min | Metrics & confusion matrix |
| 05 - Deployment | 5-10 min | SageMaker endpoint |

## 🔍 Verify Success

**Check S3**:
```bash
aws s3 ls s3://your-bucket/customer-sentiment-analysis-mlops/ --recursive
```

**Check Endpoint**:
```bash
aws sagemaker describe-endpoint \
  --endpoint-name customer-sentiment-endpoint \
  --query 'EndpointStatus'
```

## 🧪 Test Endpoint

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = runtime.invoke_endpoint(
    EndpointName='customer-sentiment-endpoint',
    ContentType='application/json',
    Body=json.dumps({'text': 'The flight was terrible and delayed!'})
)

print(json.loads(response['Body'].read()))
```

## 🐛 Common Issues

**Issue**: Workflow fails with "aws-region not supplied"
**Fix**: Region is hardcoded to `us-east-1` in all workflows

**Issue**: "AssumeRole failed"  
**Fix**: Verify trust policy has correct GitHub repo path

**Issue**: "NoSuchBucket"
**Fix**: Create S3 bucket first: `aws s3 mb s3://your-bucket`

**Issue**: Training timeout
**Fix**: Normal for CPU training. Increase timeout in workflow if needed.

## 💰 Cost Warning

⚠️ **SageMaker Endpoint costs ~$72/month if left running**

Delete when done:
```bash
aws sagemaker delete-endpoint --endpoint-name customer-sentiment-endpoint
```

## 🎯 What Gets Created

```
S3 Bucket:
└── customer-sentiment-analysis-mlops/
    ├── data/raw/Tweets.csv (3.5 MB)
    ├── data/processed/{train,val,test}.csv
    └── models/distilbert/
        ├── model files (250 MB)
        ├── model.tar.gz (for SageMaker)
        └── metrics.json

SageMaker:
├── Model: customer-sentiment-analysis-mlops-model-TIMESTAMP
├── Endpoint Config: customer-sentiment-analysis-mlops-config-TIMESTAMP  
└── Endpoint: customer-sentiment-endpoint (ml.m5.large)
```

## ✅ Success Checklist

- [ ] Repository created on GitHub
- [ ] Files uploaded via Codespaces
- [ ] GitHub secrets/variables configured
- [ ] AWS OIDC provider created
- [ ] IAM roles created with correct trust policies
- [ ] S3 bucket created
- [ ] All 5 workflows completed successfully
- [ ] Endpoint status shows "InService"
- [ ] Test inference works

---

**Need help?** Check the main README.md or open an issue on GitHub.
