<div align="center">

# ✈️ Airline Sentiment Intelligence Platform

### A production-grade MLOps pipeline that automatically trains, evaluates, and deploys a real-time sentiment analysis engine for airline brand reputation monitoring

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Model](https://img.shields.io/badge/Model-DistilBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/distilbert-base-uncased)
[![Cloud](https://img.shields.io/badge/Cloud-AWS%20SageMaker-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## 📌 Table of Contents

- [The Problem This Solves](#-the-problem-this-solves)
- [Key Outcomes](#-key-outcomes-at-a-glance)
- [Live Demo](#-live-demo)
- [System Architecture](#️-system-architecture)
- [MLOps Pipeline — 5 Stages](#-mlops-pipeline--5-stages)
- [Model Details](#-model-details)
- [Key Design Decisions](#-key-design-decisions)
- [Web Application](#-web-application)
- [Performance Results](#-performance-results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Configuration Reference](#️-configuration-reference)
- [AWS Cost Estimate](#-aws-cost-estimate)
- [Roadmap](#️-roadmap)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 The Problem This Solves

Airlines receive thousands of pieces of customer feedback every day — across Twitter, app reviews, post-flight surveys, and contact centre logs. A service failure — a mishandled bag, a delayed flight, a poor crew interaction — can go viral within minutes. Yet most airline operations teams are running on **batch pipelines that surface insights 6–24 hours after the fact.** By then, the window for service recovery has closed and the brand damage is already spreading.

This platform closes that gap. It is a fully automated, end-to-end machine learning system that transforms raw customer text into **structured sentiment intelligence in under 5 minutes** — and redeploys itself without manual intervention whenever the model or code changes.

---

## ⚡ Key Outcomes at a Glance

| Metric | Industry Baseline (no MLOps) | This Platform |
|---|---|---|
| Data-to-insight latency | 6–24 hours (batch) | < 5 minutes |
| Model deployment cycle | 6–9 months (manual handoffs) | Fully automated on every push |
| Manual engineering effort | 70–80% of team time on ETL | End-to-end pipeline automation |
| Rollback capability | Hours of manual intervention | Versioned artifacts, re-deploy in minutes |
| Model accuracy (test set) | — | ~80% accuracy, 0.78 weighted F1 |
| Inference latency | — | ~200ms per prediction |

---

## 🖥️ Live Demo

> A clean, responsive web application for both single-text analysis and bulk CSV processing.

<!-- 📸 Add a screenshot or screen-recording GIF here for maximum impact -->
<!-- Recommended tools: Kap (macOS), ShareX (Windows), or Loom for a 30-second demo -->

```
┌─────────────────────────────────────────────────────────┐
│  📝 Single Text Analysis                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ "The flight was delayed 4 hours and nobody at     │  │
│  │  the gate gave us any information."               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Result: 😞 NEGATIVE  •  Confidence: 94%               │
│  ████████████████████░░░  Negative: 0.94               │
│  ████░░░░░░░░░░░░░░░░░░░  Neutral:  0.04               │
│  ██░░░░░░░░░░░░░░░░░░░░░  Positive: 0.02               │
└─────────────────────────────────────────────────────────┘
```

**Run locally in 60 seconds:**
```bash
git clone https://github.com/<your-username>/airline-sentiment-mlops
cd airline-sentiment-mlops/sentiment-app
cp .env.example .env        # fill in your AWS credentials
docker-compose up -d

# Web UI  → http://localhost:5000
# API     → http://localhost:8000
# Docs    → http://localhost:8000/docs
```

---

## 🏗️ System Architecture

The platform is composed of two layers: an **automated MLOps pipeline** that manages the full model lifecycle, and a **serving layer** that exposes predictions via REST API and web interface.

```
┌─────────────────────────── GitHub Actions CI/CD ───────────────────────────────┐
│                                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────┐    ┌──────────┐          │
│  │  01       │    │  02       │    │  03              │    │  04       │         │
│  │  Data     │───▶│  Pre-     │───▶│  Train           │───▶│  Evaluate │         │
│  │  Download │    │  process  │    │  (SageMaker GPU) │    │  + Report │         │
│  └──────────┘    └──────────┘    └─────────────────┘    └─────┬────┘          │
│                                                                  │              │
│                                                       ┌──────────▼────────┐    │
│                                                       │  05                │    │
│                                                       │  Deploy Endpoint   │    │
│                                                       └───────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼  All data, models, and evaluation artifacts
          ┌───────────────────────────┐
          │         Amazon S3         │
          │  ├── data/raw/            │
          │  ├── data/processed/      │
          │  ├── models/              │
          │  └── evaluation/          │
          └────────────┬──────────────┘
                       │
          ┌────────────▼──────────────┐
          │   SageMaker REST Endpoint │  ←  live inference, ~200ms/request
          │   ml.m5.large (CPU)       │
          └────────────┬──────────────┘
                       │
          ┌────────────▼──────────────┐
          │      Web Application      │
          │   Flask UI ↔ FastAPI      │
          │   port 5000   port 8000   │
          └───────────────────────────┘
```

**AWS services used:** S3 (data lake + model registry), SageMaker (training + endpoint hosting), IAM (OIDC-based GitHub Actions authentication — no static credentials stored anywhere)

---

## 🔄 MLOps Pipeline — 5 Stages

Five sequentially chained GitHub Actions workflows form the backbone of this platform. Each workflow auto-triggers the next on success. Every stage also supports `workflow_dispatch` for independent re-runs.

---

### `01` — Data Ingestion

Downloads the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (14,640 tweets from 6 US carriers) from Kaggle via `kagglehub` and writes the raw CSV to Amazon S3 as the entry point into the data lake.

**Trigger:** Push to `main` on `src/data_download.py`, or manual.  
**Output:** `s3://<bucket>/<project>/data/raw/Tweets.csv`

---

### `02` — Data Preprocessing

Reads the raw CSV from S3, removes duplicates, selects relevant columns, and maps sentiment labels to integers (`negative → 0`, `neutral → 1`, `positive → 2`). Produces a **stratified 70/10/20 train-val-test split.**

**Why stratified?** The dataset is heavily skewed — 63% negative, 21% neutral, 16% positive. Stratified splitting ensures every split preserves that exact distribution, so evaluation metrics reflect true generalisability rather than split luck.

**Trigger:** Automatically after `01` completes successfully.  
**Output:** `data/processed/train.csv`, `val.csv`, `test.csv` → S3

---

### `03` — Model Training (SageMaker)

The compute-intensive core of the pipeline:

1. Packages `model_train.py` with a pinned `requirements.txt` into a `sourcedir.tar.gz` and uploads it to S3
2. Submits a **SageMaker Training Job** using the official PyTorch 2.3.0 GPU Deep Learning Container
3. Runs on **`ml.g4dn.xlarge`** (NVIDIA T4 GPU) — chosen for the right balance of GPU power and training cost
4. Fine-tunes `distilbert-base-uncased` for 3-class sequence classification using a custom **`WeightedTrainer`** subclass
5. Computes per-class weights from label distribution and applies them via `CrossEntropyLoss` to counteract the 63% negative-class bias
6. Best checkpoint is selected by **weighted F1-score** (not accuracy — see design decisions)
7. SageMaker auto-archives the trained model as `model.tar.gz` and uploads it to S3

**Configurable at runtime** (via `workflow_dispatch` inputs):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `epochs` | `10` | More epochs → longer training, potentially higher F1 |
| `batch_size` | `32` | Reduce to `16` if hitting GPU memory limits |

**Trigger:** Automatically after `02`, or manually with custom hyperparameters.  
**Output:** `s3://<bucket>/<project>/models/.../output/model.tar.gz`

---

### `04` — Model Evaluation

Downloads the latest `model.tar.gz` from S3, loads the model and tokenizer, and runs full inference on the held-out test set.

**Generates:**
- Accuracy, Precision, Recall, F1-Score (weighted and per-class)
- Full classification report (logged to GitHub Actions job summary)
- Confusion matrix visualisation — seaborn heatmap, saved as PNG
- All artifacts uploaded to `s3://<bucket>/<project>/evaluation/`

**Trigger:** Automatically after `03` completes.  
**Output:** `evaluation_results.json` + `confusion_matrix.png` → S3

---

### `05` — Model Deployment

Converts the trained model artifact into a live, callable REST endpoint:

1. Downloads and extracts `model.tar.gz` from S3
2. **Injects** a custom `inference.py` (implementing SageMaker's `model_fn` / `input_fn` / `predict_fn` / `output_fn` interface) directly into the archive — this is the key step that makes any arbitrary PyTorch/Transformers model SageMaker-servable without modification
3. Re-packages with an updated `requirements.txt` and uploads the deployment artifact to S3
4. Creates a **SageMaker Model** using the PyTorch 2.3.0 CPU inference container
5. Creates an **Endpoint Configuration** targeting `ml.m5.large`
6. Creates the endpoint if new, or performs a **graceful in-place update** if one already exists — handles `Failed` state endpoints by deleting and recreating cleanly
7. Polls until `InService` status, then **live smoke-tests** the endpoint with a real sentence

**Trigger:** Automatically after `04` completes.

**Endpoint response format:**
```json
{
  "prediction": "negative",
  "confidence": 0.94,
  "probabilities": {
    "negative": 0.94,
    "neutral": 0.04,
    "positive": 0.02
  }
}
```

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Base architecture | `distilbert-base-uncased` |
| Parameters | 66 million |
| Task | 3-class sequence classification |
| Training compute | `ml.g4dn.xlarge` — NVIDIA T4 GPU |
| Serving compute | `ml.m5.large` — CPU (cost-optimised) |
| Max input length | 128 tokens |
| Loss function | Class-weighted CrossEntropyLoss |
| Best model criterion | Weighted F1-score |
| Average inference latency | ~200ms |
| PyTorch version | 2.3.1 |
| Transformers version | 4.44.2 |

---

## 💡 Key Design Decisions

This section documents the reasoning behind the major technical choices — the thinking that separates a production system from a research notebook.

**1. DistilBERT over BERT or a larger model**  
DistilBERT retains ~97% of BERT's language understanding at 40% fewer parameters and roughly 2× faster inference. Airline tweets average ~15 words — well within the range where DistilBERT is essentially indistinguishable from its larger counterparts. It also fits comfortably on a CPU inference instance (`ml.m5.large`), eliminating the need for a GPU endpoint and reducing serving costs by ~8–10×.

**2. Class-weighted loss over resampling**  
The dataset is 63% negative. Oversampling minority classes (neutral, positive) introduces repeated training examples and risks overfitting. Undersampling throws away data. Computing inverse-frequency class weights and passing them directly into the loss function is mathematically equivalent to resampling — but cleaner, faster, and doesn't alter the underlying data distribution.

**3. F1-score as the best-model criterion, not accuracy**  
On a skewed dataset, accuracy is a misleading metric. A model that predicts "negative" for every single input would achieve 63% accuracy while being completely useless. Weighted F1-score penalises poor performance on minority classes proportionally, making it an honest measure of real-world utility.

**4. GitHub Actions over a dedicated MLOps platform**  
For a team of this size and scope, GitHub Actions provides 80% of the value at a fraction of the operational overhead of Kubeflow, Prefect, or a full MLflow stack. Code, data lineage, model artifacts, CI/CD, and deployment are all traceable in a single place — with zero additional infrastructure to manage. A dedicated platform becomes the right choice at 10+ data scientists running hundreds of parallel experiments.

**5. Decoupled training and serving compute**  
Training requires burst GPU compute for short durations. Serving requires consistent, low-latency CPU availability. Using the same instance type for both would mean either paying for idle GPU time during inference or under-powering training. Decoupling the two lets each be right-sized and billed independently.

**6. Inference script injected at deploy time, not during training**  
`model_train.py` is focused solely on producing the best possible model weights. The inference handler — input parsing, output formatting, error handling, response schema — is a deployment concern. By injecting it at deploy time, it can be updated, versioned, or swapped without triggering a full GPU retraining run.

---

## 🌐 Web Application

The `sentiment-app/` directory is a fully containerised application that sits in front of the SageMaker endpoint and makes it accessible via both a web UI and a REST API.

### Service Architecture

```
Browser
  │
  ▼
Flask Web Server  (port 5000)  ←  renders HTML templates
  │  internal HTTP
  ▼
FastAPI Backend   (port 8000)  ←  handles prediction routing + response formatting
  │  boto3 / AWS SDK
  ▼
SageMaker Endpoint             ←  DistilBERT inference
```

### Feature Summary

| Capability | Details |
|---|---|
| Single text prediction | Label + confidence + full probability breakdown + human interpretation |
| Batch / CSV upload | Upload a CSV → enriched results with predictions + summary dashboard |
| CSV auto-detection | Accepts columns named: `text`, `tweet`, `message`, `content`, `review`, `comment` |
| REST API | `POST /predict`, `POST /predict/batch`, `POST /predict/csv`, `GET /health` |
| Interactive API docs | Auto-generated Swagger UI at `/docs` |
| Containerised deployment | Both services via single `docker-compose up` |

### API Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely love the new boarding experience, very smooth!"}'

# Response
{
  "prediction": "positive",
  "confidence": 0.91,
  "probabilities": { "positive": 0.91, "neutral": 0.06, "negative": 0.03 },
  "sentiment_emoji": "😊",
  "interpretation": "This text expresses very strong positive sentiment..."
}

# Batch (up to 100 texts)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["Great flight!", "Nothing special", "Lost my bag again"]}'

# CSV upload
curl -X POST http://localhost:8000/predict/csv -F "file=@feedback.csv"
```

---

## 📊 Performance Results

| Metric | Test Set Score |
|---|---|
| Accuracy | ~80% |
| Weighted F1-Score | ~0.78 |
| Weighted Precision | ~0.79 |
| Weighted Recall | ~0.80 |
| Inference latency | ~200ms/request |

The confusion matrix and full per-class classification report are generated automatically during pipeline Stage 4 and rendered directly in the GitHub Actions job summary after each training run. Artifacts are also saved to S3 for audit history.

---

## 📁 Repository Structure

```
airline-sentiment-mlops/
│
├── .github/
│   └── workflows/
│       ├── 01-data-download.yml     # Kaggle download → S3 upload
│       ├── 02-data-preprocess.yml   # Clean, encode labels, stratified split
│       ├── 03-model-train.yml       # SageMaker GPU training job
│       ├── 04-model-eval.yml        # Test set metrics + confusion matrix
│       └── 05-model-deploy.yml      # Endpoint creation + smoke test
│
├── src/
│   ├── data_download.py             # Dataset ingestion script
│   ├── data_preprocess.py           # Feature prep + train/val/test split
│   ├── model_train.py               # DistilBERT fine-tuning (SageMaker-compatible)
│   ├── model_eval.py                # Test set evaluation + visualisations
│   └── model_deploy.py              # Deployment packaging + endpoint management
│
└── sentiment-app/
    ├── backend/
    │   ├── api.py                   # FastAPI — prediction endpoints (port 8000)
    │   └── app.py                   # Flask — web UI server (port 5000)
    ├── frontend/
    │   ├── templates/
    │   │   ├── index.html           # Single text prediction page
    │   │   └── batch.html           # CSV batch upload page
    │   └── static/
    │       ├── css/style.css
    │       └── js/
    │           ├── app.js           # Single prediction logic
    │           └── batch.js         # Batch/CSV processing logic
    ├── Dockerfile.api               # FastAPI container
    ├── Dockerfile.web               # Flask container
    ├── docker-compose.yml           # Orchestrates both services
    └── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- AWS account with permissions for S3, SageMaker, and IAM
- GitHub repository with Actions enabled
- Docker and Docker Compose (for the web application)

### Step 1 — Configure GitHub Repository Secrets & Variables

Navigate to: **Settings → Secrets and Variables → Actions**

**Repository Secrets:**
| Secret | Description |
|---|---|
| `AWS_ROLE_ARN` | IAM role ARN trusted by GitHub OIDC |
| `SAGEMAKER_ROLE_ARN` | IAM role ARN for SageMaker training jobs |
| `KAGGLE_USERNAME` | Kaggle account username |
| `KAGGLE_KEY` | Kaggle API key |

**Repository Variables:**
| Variable | Example Value |
|---|---|
| `S3_BUCKET` | `my-mlops-bucket` |
| `AWS_REGION` | e.g. `us-east-1` |
| `PROJECT_NAME` | `airline-sentiment` |
| `ENDPOINT_NAME` | `sentiment-analysis-endpoint` |

> This project uses **GitHub OIDC** for AWS authentication. No static access keys are stored in GitHub Secrets — GitHub assumes an IAM role directly, with session-scoped credentials.

### Step 2 — Trigger the Full Pipeline

Go to **Actions → `01 - Data Download` → Run workflow.**

All subsequent stages chain automatically:
```
01 (data) → 02 (preprocess) → 03 (train, ~45 min) → 04 (evaluate) → 05 (deploy)
```

Total time from trigger to live endpoint: **~50–70 minutes**, dominated by the SageMaker training job.

### Step 3 — Launch the Web Application

```bash
cd sentiment-app

# Set up environment
cat > .env << EOF
SAGEMAKER_ENDPOINT=<your-endpoint-name>
AWS_REGION=<your-region>
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
EOF

# Start both services
docker-compose up -d

# Stop when done
docker-compose down
```

### Re-running Individual Stages

Every workflow supports `workflow_dispatch`, so any stage can run in isolation:

```bash
# Re-evaluate without retraining    → trigger: 04 - Model Evaluation
# Re-deploy to a new endpoint        → trigger: 05 - Model Deployment
# Retrain with different epochs      → trigger: 03 - Model Training (set inputs)
```

---

## ⚙️ Configuration Reference

### Web App Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SAGEMAKER_ENDPOINT` | Name of the deployed SageMaker endpoint | `sentiment-analysis-endpoint` |
| `AWS_REGION` | AWS region where endpoint is hosted (e.g. `us-east-1`) | — |
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |
| `API_URL` | Internal FastAPI URL consumed by Flask | `http://localhost:8000` |

### Training Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `epochs` | `10` | Configurable at runtime via `workflow_dispatch` |
| `batch_size` | `32` | Reduce to `16` if hitting GPU memory errors |
| `max_length` | `128` | Token sequence length — sufficient for tweet-length text |
| `warmup_steps` | `500` | Linear warmup before learning rate decay |
| `weight_decay` | `0.01` | L2 regularisation |

---

## 💰 AWS Cost Estimate

| Resource | Assumed Usage | Estimated Cost |
|---|---|---|
| SageMaker Training (`ml.g4dn.xlarge`) | 1 run/day | ~$0.74 per run |
| SageMaker Endpoint (`ml.m5.large`) | 8 hrs/day active | ~$0.92/day |
| Amazon S3 | ~500 MB (data + model) | < $1/month |
| GitHub Actions | Within free tier | Free |
| **Estimated total (active development)** | | **~$30–40/month** |

> ⚠️ **Important:** The SageMaker endpoint accrues charges every hour it is running. Delete it when not needed:
> ```bash
> aws sagemaker delete-endpoint --endpoint-name <your-endpoint-name>
> ```

---

## 🗺️ Roadmap

The current platform handles English-language airline tweets. Planned extensions include:

- [ ] **Multilingual support** — extend to Hindi, Mandarin, and Spanish using `xlm-roberta-base`
- [ ] **Drift detection** — integrate Population Stability Index (PSI) monitoring; auto-trigger retraining when input distribution shifts beyond threshold
- [ ] **Streaming ingestion** — replace batch S3 uploads with real-time Kafka/Amazon MSK consumers for sub-minute latency
- [ ] **A/B model testing** — canary deployments routing a configurable percentage of traffic to a challenger model, with automatic rollback on metric degradation
- [ ] **Explainability layer** — integrate SHAP or attention visualisations into the web UI for transparent predictions
- [ ] **Alerting** — Slack/PagerDuty notifications triggered when negative sentiment on a specific route spikes above a defined threshold
- [ ] **User authentication** — API key management and login for the web application

---

## 🔒 Security

- **No credentials in code** — AWS access uses GitHub OIDC role assumption; no static keys in any workflow file or repository
- **`.env` is gitignored** — never committed; a `.env.example` template is provided for local setup
- **IAM least privilege** — the GitHub Actions role and SageMaker role carry only the permissions each task requires
- **All secrets in GitHub Secrets** — not in environment variables, workflow files, or repository variables
- **For production hardening:** enable HTTPS, configure CORS to specific origins, add rate limiting, sanitise and validate all inputs, and manage credentials via AWS Secrets Manager

---

## 🐛 Troubleshooting

**Endpoint not found:**
```bash
aws sagemaker describe-endpoint \
  --endpoint-name <your-endpoint-name> \
  --region <your-region>
```

**Training job failed:**
```bash
# Get the failure reason
aws sagemaker describe-training-job \
  --training-job-name <job-name> \
  --query 'FailureReason'

# Tail full logs from CloudWatch
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

**Processed data not found (training fails at data load):**
```bash
# Confirm Stage 02 ran successfully first
aws s3 ls s3://<your-bucket>/<your-project>/data/processed/
# Should list: train.csv, val.csv, test.csv
```

**AWS credentials error in Docker:**
```bash
docker-compose logs api    # check for auth or endpoint errors
# Verify .env — no extra spaces, no trailing quotes
```

**Port already in use:**
```bash
# Edit docker-compose.yml to use alternate ports
ports:
  - "5001:5000"   # Web UI
  - "8001:8000"   # API
```

---

## 📚 References

- [DistilBERT: a distilled version of BERT](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) — CrowdFlower via Kaggle
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [GitHub Actions OIDC with AWS](https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit with a descriptive message
4. Open a Pull Request with context on what was changed and why

For significant changes, open an issue first to discuss the approach.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Built to demonstrate what a production MLOps system looks like — from raw data ingestion to a live, smoke-tested endpoint, with zero manual steps in between.*

</div>
