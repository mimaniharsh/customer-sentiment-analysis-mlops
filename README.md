<div align="center">

<br/>

# SentiAI — Customer Intelligence Platform

### Production-grade NLP · Multi-model comparative analysis · End-to-end MLOps on AWS

<br/>

[![CI](https://img.shields.io/github/actions/workflow/status/mimaniharsh/customer-sentiment-analysis-mlops/01-ci.yml?label=CI&style=flat-square&logo=github&logoColor=white)](https://github.com/mimaniharsh/customer-sentiment-analysis-mlops/actions/workflows/01-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15.0.3-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/distilbert-base-uncased)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)

<br/>

> **TAPMI MBA–AI&DS · Concurrent Running Project (CRP) · ADS 5711 · Term 3 · Batch 2025–2027**
>
>

<br/>

[Overview](#-overview) · [The Problem](#-the-problem) · [Model Study](#-comparative-model-study) · [Architecture](#-platform-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [ML Pipeline](#-ml-training-pipeline) · [Dashboard](#-executive-dashboard) · [MLOps](#-mlops--cicd) · [Security](#-security-model)

</div>

---

## Overview

SentiAI is a full-stack, multi-tenant customer intelligence platform that transforms raw social media feedback into structured, actionable business intelligence. Built as the Term 3 deliverable of the Concurrent Running Project at TAPMI, it extends the comparative NLP model study conducted in Term 2 into a production-ready cloud system deployed on Amazon Web Services.

The platform operationalises three sequentially developed NLP architectures — TF-IDF + Logistic Regression, Bidirectional LSTM, and fine-tuned DistilBERT — through a rigorous comparative evaluation on the Twitter US Airline Sentiment dataset. **DistilBERT is selected as the sole production model** based on empirical performance across accuracy, neutral-class F1, and sarcasm handling. It is trained on AWS SageMaker (`ml.g4dn.xlarge`) and served through a managed inference endpoint (`ml.m5.large`), backing a FastAPI REST layer, a Next.js executive dashboard, and a suite of automated MLOps workflows.

Every prediction is enriched by a four-head post-processing pipeline — sarcasm detection, severity classification, attribute tagging, and root-cause decomposition — delivering not a bare sentiment label, but an intelligence card with urgency tier, business impact score, escalation recommendation, and strategic insight, all within 200 milliseconds.

---

## The Problem

Airlines and service-intensive organisations operate simultaneously in two distinct markets: one where revenue is generated, and one where brand perception is formed. The second market — social media — is noisy, fast-moving, and structurally adversarial to standard keyword-based sentiment tools.

The core failure mode is sarcasm.

> *"Oh brilliant, another delay. Truly phenomenal service."*

A conventional sentiment system assigns this tweet a **positive** classification. The customer is, in reality, furious — a high churn risk whose signal has been entirely missed. At scale, across thousands of daily tweets, this systematic misclassification produces suppressed escalation rates, unrecovered customers, and brand health metrics that bear no relationship to operational reality.

SentiAI addresses this directly. It identifies sarcasm through a multi-signal probabilistic model, applies a label override when warranted, classifies the severity of the underlying complaint, decomposes it by root cause, and recommends a specific escalation action — all within a single API call completing in under 200 milliseconds.

---

## Comparative Model Study

The academic foundation of this project is a disciplined three-phase comparative study. Each model was trained and evaluated on identical data splits, with results directly informing the decision to deploy DistilBERT to production.

| Dimension | TF-IDF + Logistic Regression | Bidirectional LSTM | DistilBERT |
|---|:---:|:---:|:---:|
| **Approach** | Bag-of-words · linear classifier | Sequence model · trainable embeddings | Pre-trained transformer · fine-tuned |
| **Parameters** | Sparse (vocab × 3 class weights) | ~2.5 M | 66 M |
| **Test Accuracy** | ~72% | ~76% | **~91%** |
| **Weighted F1** | ~0.79 | ~0.84 | **~0.91** |
| **Neutral-Class F1** | 0.44 | 0.60 | **0.82** |
| **Inference Latency** | < 5 ms | ~15 ms | ~180 ms |
| **Interpretability** | High — coefficient weights | Medium — attention proxy | High — token saliency |
| **Sarcasm Handling** | Fails — context-blind | Partial — limited sequence range | Strong — bidirectional full context |
| **Training Infrastructure** | GitHub Actions runner (CPU) | Google Colab (GPU) | AWS SageMaker `ml.g4dn.xlarge` |
| **Cloud Deployment** | — | — | ✦ **Production endpoint** |

> Metrics are logged to Amazon S3 and surfaced in the GitHub Actions workflow summary after every training run.

### The Decision Rationale

**Why DistilBERT was selected for production deployment:**

Bidirectional contextual encoding is not an incremental improvement over sequential or bag-of-words approaches — it is architecturally superior for the specific failure modes present in this dataset. Sarcasm, negation, and semantic ambiguity require a model to evaluate each token against its full surrounding context simultaneously in both directions. DistilBERT's self-attention mechanism achieves this natively, reducing neutral-class misclassification by 38 percentage points relative to the TF-IDF baseline.

At 66 million parameters — 40% fewer than BERT-base — DistilBERT retains 97% of BERT's representational quality at a materially lower inference cost, making it viable for real-time API serving on a CPU-class endpoint (`ml.m5.large`).

**Why TF-IDF and LSTM remain in the comparative framework:**

When a classification decision must be explained to a non-technical stakeholder, a compliance team, or a regulatory body, logistic regression coefficient weights are unambiguous and interpretable in a way that transformer attention maps are not without supplementary tooling. The comparative architecture preserves both prior models as auditable benchmarks and maintains the full academic lineage of the project across Term 2 and Term 3.

---

## Platform Architecture

SentiAI is architected as a multi-tenant SaaS platform spanning six containerised services, a three-stage CI/CD pipeline, and a managed cloud ML inference layer. The entire local development stack launches with a single command.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud — Production                               │
│                                                                                │
│   ┌───────────────────────┐     ┌──────────────────────────────────────────┐   │
│   │  Next.js 15.0.3       │     │  FastAPI 0.115  ·  Uvicorn 0.30.6        │   │
│   │  TypeScript 5         │◀──▶│  Python 3.11   ·  2 async workers        │   │
│   │  Executive Dashboard  │     │  JWT (RS256)   ·  RBAC — 4 roles         │   │
│   │  7 stakeholder views  │     │  SlowAPI rate limiting · GZip middleware │   │
│   └───────────────────────┘     └─────────────────────┬────────────────────┘   │
│                                                       │                        │
│                                     ┌─────────────────▼────────────────┐       │
│                                     │   AWS SageMaker Endpoint         │       │
│                                     │   DistilBERT (fine-tuned)        │       │
│                                     │   Instance:  ml.m5.large (CPU)   │       │
│                                     │   Latency:   ~180 ms             │       │
│                                     │   Batch:     up to 500 texts     │       │
│                                     └──────────────────────────────────┘       │
│                                                                                │
│   ┌──────────────────────────┐    ┌──────────────────────────────────────┐     │
│   │  PostgreSQL 16 Alpine    │    │  Redis 7 Alpine  ·  Celery 5.4       │     │
│   │  7 async ORM models      │    │  Worker: concurrency=4               │     │
│   │  Row-level tenant ISO    │    │  Beat: hourly metrics                │     │
│   │  Alembic migrations      │    │        drift detection at 02:00 UTC  │     │
│   └──────────────────────────┘    └──────────────────────────────────────┘     │
│                                                                                │
│  S3           Model artifacts · processed data · evaluation outputs            │
│  ECR          Docker images tagged by Git SHA — immutable deployments          │
│  ECS Fargate  API + Web containers — rolling zero-downtime updates             │
│  Secrets Mgr  All production credentials fetched at runtime via IAM role       │
│  OIDC         GitHub Actions auth — zero long-lived access keys                │
│  Prometheus   API latency · endpoint health · prediction class distribution    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Service Inventory

| Service | Technology | Port | Responsibility |
|---|---|:---:|---|
| `api` | FastAPI 0.115.0 · Uvicorn 0.30.6 · Python 3.11 | 8000 | Core ML API — prediction, analytics, reporting, authentication |
| `web` | Next.js 15.0.3 · React 18 · TypeScript 5 · Tailwind CSS | 3000 | Executive dashboard — 7 audience-specific pages |
| `postgres` | PostgreSQL 16 Alpine | 5432 | Primary datastore — 7 tables, row-level tenant isolation, async ORM |
| `redis` | Redis 7 Alpine | 6379 | Celery broker (DB 1) and API response cache (DB 0) |
| `worker` | Celery 5.4.0 · Python 3.11 | — | Background tasks — drift detection, metric aggregation, report generation |
| `beat` | Celery Beat scheduler | — | Cron orchestration — hourly snapshots, daily drift detection at 02:00 UTC |
| `prometheus` | Prometheus 2.54.1 | 9090 | Latency histograms, endpoint health, prediction class distributions |

---

## The Enrichment Pipeline

Raw DistilBERT output — three class probabilities — is insufficient for operational decision-making. Every prediction passes through four enrichment heads before a response is returned.

```
  Raw Tweet Text
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │      AWS SageMaker Endpoint  ·  DistilBERT (66M)        │
  │      Instance: ml.m5.large (CPU)  ·  ~180ms latency     │
  │                                                         │
  │  Output: prob_positive · prob_neutral · prob_negative   │
  └───────────────────────────┬─────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
  ┌─────────────────┐  ┌───────────────┐  ┌──────────────────────┐
  │  Sarcasm Head   │  │ Severity Head │  │  Attribute Tagger    │
  │  5-signal model │  │ 4-tier matrix │  │  7 business domains  │
  │  Threshold 0.55 │  │ with auditable│  │  35-issue taxonomy   │
  │  Label override │  │ justification │  │  Root-cause clauses  │
  └────────┬────────┘  └───────┬───────┘  └──────────┬───────────┘
           │                   │                     │
           └───────────────────┼─────────────────────┘
                               ▼
              ┌───────────────────────────────┐
              │   Override & Risk Engine      │
              │   Sarcasm ≥ 0.55 → flip label │
              │   CRITICAL severity → flip    │
              │   Override reason in response │
              └──────────────┬────────────────┘
                             ▼
              ┌──────────────────────────────────────┐
              │          Intelligence Card           │
              │  sentiment          confidence       │
              │  sarcasm_prob       override_reason  │
              │  severity_tier      severity_score   │
              │  urgency            impact_score     │
              │  attributes         theme_breakdown  │
              │  top_tokens         strategic_insight│
              │  escalation_action  latency_ms       │
              └──────────────────────────────────────┘
```

### Sarcasm Detection — 5-Signal Probabilistic Model

| Signal | Mechanism | Weight |
|---|---|:---:|
| **Trigger + context** | 14+ trigger phrases (e.g. `"brilliant"`, `"phenomenal"`) paired with a required negative context word (e.g. `"delay"`, `"lost"`, `"broken"`). The trigger fires only when both elements are present — not on a positive word alone. | 0.35 per hit |
| **Always-sarcastic phrases** | Phrases unconditionally flagged regardless of surrounding context — e.g. `"thanks for nothing"`, `"what a surprise"`, `"you've outdone yourselves"` | 0.45 per hit |
| **Probability mismatch** | Model assigns a positive or neutral label, but `prob_negative > 0.20` — internal model uncertainty inconsistent with the surface classification | `0.50 × prob_negative` |
| **Negation flip** | Negation word immediately preceding a positive descriptor — `"not great"`, `"barely acceptable"`, `"nothing impressive"` | 0.20 |
| **Exclamation amplifier** | Exclamation mark present alongside a complaint keyword (`"delay"`, `"lost"`, `"cancelled"`, `"broken"`) | 0.15 |

### Severity Classification — 4-Tier Escalation Matrix

| Tier | Score Range | Example Triggers | Mandated Escalation |
|---|:---:|---|---|
| **CRITICAL** | 0.75 – 1.00 | cancel, refund, fraud, stranded, data breach, never again | Senior leadership · within 1 hour · executive documentation required |
| **HIGH** | 0.50 – 0.75 | delay, broken, terrible, unresolved, failure, disaster | Dedicated team · within 4 hours · weekly review flag |
| **MEDIUM** | 0.25 – 0.50 | slow, confusing, disappointing, mediocre | Scheduled response · within 48 hours · sprint inclusion |
| **LOW** | 0.00 – 0.25 | *(no negative signals detected)* | Log for trend analysis · monthly sentiment report |

---

## Repository Structure

```
customer-sentiment-analysis-mlops/
│
├── apps/
│   │
│   ├── api/                                   # Python 3.11 · FastAPI backend
│   │   ├── app/
│   │   │   ├── api/v1/endpoints/              # predict · batch · csv · analytics
│   │   │   │                                  # reports · auth · tenants · health
│   │   │   ├── ml/
│   │   │   │   └── pipeline.py                # 4-head enrichment engine (~920 lines)
│   │   │   ├── models/
│   │   │   │   └── db.py                      # 7 async SQLAlchemy models
│   │   │   ├── core/                          # config · database · security · deps
│   │   │   └── workers/                       # Celery — drift · metrics · reports
│   │   ├── migrations/                        # Alembic migration scripts
│   │   ├── tests/                             # pytest async test suite (httpx)
│   │   ├── requirements.txt
│   │   └── Dockerfile                         # Two-stage build · non-root runtime user
│   │
│   └── web/                                   # TypeScript · Next.js 15 · Node 20
│       └── src/app/
│           ├── page.tsx                       # Overview — health score, NPS trend, alerts
│           ├── analytics/                     # Sentiment timeseries · attribute heatmap
│           ├── clusters/                      # Root-cause topic groups · negative rates
│           ├── demo/                          # Live prediction stream · IntelligenceCard
│           ├── reports/                       # 1-click report generation
│           ├── strategy/                      # Business impact simulator
│           └── settings/                      # Tenant config · API key management
│
├── packages/ml/                               # Python 3.11 · SageMaker pipeline
│   ├── pipeline/
│   │   ├── data_download.py                   # Kaggle download via kagglehub
│   │   └── data_preprocess.py                 # Text cleaning · stratified splits · S3 upload
│   ├── training/
│   │   └── model_train.py                     # WeightedTrainer · DistilBERT fine-tuning
│   │                                          # 10 documented engineering improvements
│   ├── evaluation/
│   │   └── model_eval.py                      # Per-class F1 · confusion matrix · ECE
│   └── serving/
│       └── model_deploy.py                    # SageMaker endpoint create / blue-green update
│
├── .github/workflows/
│   ├── 01-ci.yml                              # Secret scan → lint → test → build
│   ├── 02-deploy.yml                          # ECS rolling deploy via OIDC
│   └── 03-ml-train-deploy.yml                 # Full SageMaker pipeline (manual trigger)
│
├── scripts/
│   └── demo_live_stream.py                    # Live demonstration entry point
│
├── infra/docker/
│   ├── nginx.conf
│   └── prometheus.yml
│
├── docker-compose.yml                         # 6-service local stack · one command
├── .env.example                               # Credential template — .env is git-ignored
└── SETUP.md                                   # Full deployment and configuration guide
```

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, an AWS account with SageMaker access, and Kaggle API credentials.

```bash
# 1. Clone the repository
git clone https://github.com/mimaniharsh/customer-sentiment-analysis-mlops.git
cd customer-sentiment-analysis-mlops

# 2. Configure your environment
cp .env.example .env
# Open .env and populate all required values
```

**Minimum required `.env` configuration:**

```env
# AWS Infrastructure
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
S3_BUCKET=your-s3-bucket-name

# Application Security
APP_SECRET_KEY=<32-character-random-string>
JWT_SECRET_KEY=<32-character-random-string>
```

```bash
# 3. Launch all six services
docker compose up --build
# First build: approximately 4 minutes  ·  Subsequent starts: approximately 20 seconds
```

**Platform access points:**

| URL | Service |
|---|---|
| http://localhost:3000 | Executive dashboard (Next.js) |
| http://localhost:8000/docs | Swagger UI — interactive API documentation (development only) |
| http://localhost:9090 | Prometheus metrics |

**Run the live demonstration:**

```bash
pip install requests

# Stream tweets through the full pipeline in real time
python scripts/demo_live_stream.py

# Process an entire CSV file in batch mode
python scripts/demo_live_stream.py --mode batch --file Tweets.csv

# Presentation mode — 2-second pacing between predictions
python scripts/demo_live_stream.py --delay 2.0 --limit 50 --shuffle
```

---

## API Reference

### Single-Text Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Three consecutive delays this week. Absolutely unacceptable."}'
```

### Batch Prediction — up to 500 texts per request

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Fantastic flight!", "Lost my luggage again.", "Flight departed on time."]}'
```

### CSV Upload — up to 10,000 rows

```bash
curl -X POST http://localhost:8000/api/v1/predict/csv \
  -F "file=@Tweets.csv"
```

**Auto-detected column names:** `text` · `tweet` · `message` · `content` · `review`

### Complete Response Contract

```json
{
  "sentiment": "negative",
  "confidence": 0.943,
  "probabilities": {
    "positive": 0.028,
    "neutral": 0.029,
    "negative": 0.943
  },

  "sarcasm_probability": 0.82,
  "sarcasm_label": "Likely sarcastic",
  "sarcasm_signals": [
    "Trigger 'phenomenal' with required context: {'delay'} — score +0.35",
    "Probability mismatch — positive label, prob_negative 27.9% — score +0.14",
    "Exclamation mark on complaint topic — score +0.15"
  ],
  "sentiment_override_reason": "Overridden from 'positive': sarcasm probability 82% exceeds threshold of 55%.",

  "severity_tier": "HIGH",
  "severity_score": 0.61,
  "severity_criteria": ["Significant service failure — escalation warranted"],

  "urgency": "high",
  "risk": {
    "tier": "HIGH",
    "action": "Assign a dedicated team within 4 hours. Flag for weekly operations review.",
    "label": "Service recovery required"
  },

  "attributes": ["reliability", "performance"],
  "theme_breakdown": [
    {
      "clause": "three consecutive delays",
      "attribute": "reliability",
      "issue": "repeated_delay",
      "mini_sentiment": "negative"
    }
  ],

  "business_impact_score": 0.61,
  "strategic_insight": "High-severity reliability failure detected. Unresolved signals in this tier are statistically associated with elevated churn risk and adverse NPS movement.",

  "explanation": {
    "top_tokens": [
      { "token": "unacceptable", "importance": 0.41 },
      { "token": "delays",       "importance": 0.29 },
      { "token": "consecutive",  "importance": 0.17 }
    ]
  },

  "model_version": "v2",
  "endpoint_name": "customer-sentiment-analysis-mlops-endpoint",
  "latency_ms": 187
}
```

---

## ML Training Pipeline

The model training pipeline runs entirely on AWS infrastructure — no local GPU is required. It is triggered manually through GitHub Actions with configurable epoch and batch size parameters.

### Four Sequential Stages

| Stage | Infrastructure | Duration | What Occurs |
|---|---|:---:|---|
| **1 · Data Preprocessing** | GitHub Actions Ubuntu runner (CPU) | ~5 min | Dataset downloaded from Kaggle via `kagglehub` · text cleaning pipeline applied · 70/10/20 stratified split by class label · processed CSVs uploaded to S3 |
| **2 · Model Training** | AWS SageMaker `ml.g4dn.xlarge` · NVIDIA T4 GPU · 30 GB EBS volume | ~35 min | DistilBERT fine-tuned with `WeightedTrainer` · 10 documented engineering improvements · early stopping patience = 3 · model artifact and metrics saved to S3 |
| **3 · Evaluation** | GitHub Actions Ubuntu runner (CPU) | ~5 min | Batched inference on held-out test set · per-class F1, confusion matrix, and Expected Calibration Error (ECE) computed · artifacts uploaded to S3 and GitHub Actions workflow summary |
| **4 · Deployment** | GitHub Actions runner + SageMaker API | ~10 min | Model packaged with production `inference.py` · SageMaker model and endpoint configuration created or updated via blue/green swap · endpoint polled until `InService` |

**How to trigger:** GitHub Actions → *ML — Train & Deploy SageMaker Model* → *Run workflow* → configure `epochs`, `batch_size`, and `deploy_after_train`.

### Training Configuration

| Hyperparameter | Value | Engineering Rationale |
|---|---|---|
| Base model | `distilbert-base-uncased` | 66M parameters · 97% of BERT-base accuracy at 60% of inference cost |
| Training instance | `ml.g4dn.xlarge` — NVIDIA T4 GPU | ~$0.43 per run at approximately 35 minutes |
| Inference instance | `ml.m5.large` — CPU | ~$83/month while the endpoint is active |
| Max epochs | 10 with early stopping (patience = 3) | Typical stop at epoch 5–7; saves ~$0.30–0.50 per run in GPU-hours |
| Effective batch size | 32 × gradient accumulation 2 = 64 | Larger effective batch stabilises training on a 63% negative-class-skewed dataset |
| LR scheduler | Cosine decay | Smoother terminal-epoch convergence versus linear decay |
| Warmup ratio | 0.06 of total steps | Scales proportionally with dataset size; avoids instability from a fixed step count |
| Label smoothing | 0.10 | Prevents overconfident softmax output; aligns displayed confidence with empirical accuracy |
| Max sequence length | 128 tokens | Optimal for tweet-length text; configurable to 256 for tickets or longer feedback forms |
| Loss function | `WeightedCrossEntropy` (instance-cached) | Class weights compensate for the 63% negative imbalance; cached to avoid per-forward-pass re-instantiation overhead |
| Padding strategy | `DataCollatorWithPadding` | Dynamic per-batch padding; eliminates wasted compute from static dataset-wide padding |
| Confidence calibration | Temperature scaling T = 1.3 | Post-processes raw softmax probabilities so that a displayed confidence of 82% corresponds empirically to ~82% accuracy on held-out examples |

### Expected Performance — DistilBERT Production Model

| Metric | Value |
|---|---|
| Validation accuracy | ~91% |
| Test accuracy | ~91% |
| Weighted F1 | ~0.91 |
| F1 — Negative class | ~0.93 |
| F1 — Neutral class | ~0.82 |
| F1 — Positive class | ~0.87 |
| Expected Calibration Error (ECE) | < 0.05 (target — excellent calibration) |
| Inference latency | ~180 ms on `ml.m5.large` |
| Training cost | ~$0.43 per run (`ml.g4dn.xlarge`) |
| Endpoint cost | ~$83/month while active (`ml.m5.large`) |

---

## Executive Dashboard

Seven purpose-built views, each designed for a distinct stakeholder audience.

| Page | Route | Primary Audience | Key Components |
|---|---|---|---|
| **Overview** | `/` | Executive / Board | Brand health score · 30-day NPS trend · critical alert count · drift status indicator |
| **Analytics** | `/analytics` | Operations / Customer Experience | Sentiment timeseries · 7-category attribute heatmap · urgency distribution breakdown |
| **Clusters** | `/clusters` | Product / Engineering | Root-cause topic groups · negative rate per theme · cluster-level insights |
| **Live Demo** | `/demo` | Engineering / Sales | Real-time IntelligenceCard per tweet · bubble chart (severity × confidence) · root-cause pin pointer |
| **Reports** | `/reports` | Analysts / Management | 1-click executive summary · risk breakdown · drift analysis — async generation via Celery, persisted to S3 |
| **Strategy** | `/strategy` | Senior Leadership | Business impact simulator — model the NPS impact of resolving individual issue categories |
| **Settings** | `/settings` | Platform Administrator | Tenant configuration · API key management · model version selection |

---

## MLOps & CI/CD

### Three GitHub Actions Workflows

**`01-ci.yml` — Continuous Integration**

Triggered on every push to `main` or `develop` and on all pull requests targeting `main`. Executes three jobs in sequence: (1) **Secret scanning** — regex scan across all Python, TypeScript, JavaScript, and YAML files for hardcoded AWS access key patterns (`AKIA[0-9A-Z]{16}`); any match fails the build immediately, before compilation or testing begins. (2) **Python CI** — Ruff linter followed by pytest with async httpx test client, running on Python 3.11. (3) **Frontend CI** — Node 20 · `npm install --legacy-peer-deps` · Next.js 15 production build verification.

**`02-deploy.yml` — Continuous Deployment**

Manual trigger only, with `staging` or `production` environment selector. Authenticates to AWS via OpenID Connect (`id-token: write` permission) — no static AWS access keys are stored in GitHub Secrets. Builds and pushes API and Web Docker images to Amazon ECR, tagged by Git SHA for immutable deployment traceability. Deploys both services to ECS Fargate with a rolling update strategy — zero downtime throughout.

**`03-ml-train-deploy.yml` — ML Pipeline**

Manual trigger with `epochs`, `batch_size`, and `deploy_after_train` inputs. Executes all four pipeline stages in sequence (preprocess → train → evaluate → deploy). All evaluation outputs — `evaluation_results.json`, `confusion_matrix.png`, and `calibration_plot.png` (reliability diagram) — are uploaded to both S3 and the GitHub Actions workflow summary for persistent traceability.

### Drift Detection

Model quality is monitored continuously through two complementary statistical methods.

**Population Stability Index (PSI)** compares the recent distribution of predicted sentiment scores against a reference window established at deployment time. A PSI score exceeding 0.20 constitutes a material drift event and triggers an alert persisted to the `drift_records` table with `alert_triggered = true`.

**Kolmogorov–Smirnov (KS) statistic** measures distributional shift in raw class probability outputs, providing an independent second signal of data or model drift not captured by PSI alone.

Drift detection is executed by the Celery Beat scheduler daily at 02:00 UTC. Alerts surface on the Overview dashboard and are available for human review within the Reports section.

---

## Security Model

No credentials exist anywhere in the repository — including version history. The CI pipeline enforces this structurally: secret scanning runs as the first job in every workflow and will halt execution if any AWS key pattern is detected, before a single line of application code is compiled, tested, or deployed.

| Environment | Credential Source | Mechanism |
|---|---|---|
| **Local development** | `.env` file (git-ignored by `.gitignore`) | `pydantic-settings` reads at application startup |
| **Production (AWS)** | AWS Secrets Manager | Fetched at container startup via IAM instance role — zero static keys in code or image layers |
| **CI/CD (GitHub Actions)** | GitHub Secrets + AWS OIDC | Ephemeral role assumed per workflow run — no long-lived access keys stored anywhere |

**Additional security controls implemented:**

- JWT authentication (`python-jose` + `passlib` bcrypt) with four RBAC roles: `owner` · `admin` · `analyst` · `viewer`
- Input validation via Pydantic v2 `field_validator` — text stripped, length enforced, and sanitised at the schema boundary
- SQL injection prevention — SQLAlchemy async ORM with parameterised queries exclusively; no raw SQL strings anywhere in the codebase
- CORS configured with an explicit `allow_origins` list in production; wildcard is never used
- `RedactingFilter` on all structured log output — strips credential patterns before any log is emitted
- Per-IP rate limiting via SlowAPI with configurable requests-per-minute thresholds

---

## Cost Profile

| Resource | Specification | Estimated Cost |
|---|---|---|
| SageMaker training | `ml.g4dn.xlarge` · NVIDIA T4 GPU · 30 GB EBS | ~$0.43 per run |
| SageMaker inference | `ml.m5.large` · managed endpoint | ~$83/month while active |
| ECS Fargate | API + Web containers | ~$20–30/month |
| PostgreSQL | `db.t3.micro` or equivalent EC2 | ~$15–25/month |
| Amazon S3 | Models · datasets · reports | < $5/month |
| **Total — endpoint active** | | **~$120–150/month** |
| **Total — endpoint deleted** | Redeploy on demand in ~10 minutes | **~$40–60/month** |

**Recommended practice — delete the SageMaker endpoint between active development sessions:**

```bash
aws sagemaker delete-endpoint \
  --endpoint-name customer-sentiment-analysis-mlops-endpoint \
  --region ap-south-1

# Restore at any time: GitHub Actions → ML — Train & Deploy → Run workflow
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Languages** | Python 3.11 (API · ML pipeline) · TypeScript 5 (frontend) |
| **NLP / ML** | HuggingFace Transformers 4.44.2 · PyTorch 2.3.1 · scikit-learn 1.5.1 |
| **API Framework** | FastAPI 0.115.0 · Uvicorn 0.30.6 · Pydantic v2.9.2 · SQLAlchemy 2.0 async |
| **Task Queue** | Celery 5.4.0 · Redis 7 (broker + cache) · asyncpg 0.29.0 |
| **Database** | PostgreSQL 16 Alpine · Alembic 1.13.3 (migrations) |
| **Frontend** | Next.js 15.0.3 · React 18 · Tailwind CSS · Recharts · TanStack Query v5 |
| **Cloud — Compute** | SageMaker `ml.g4dn.xlarge` (training) · `ml.m5.large` (inference) · ECS Fargate |
| **Cloud — Storage** | Amazon S3 (artifacts, datasets, reports) · Amazon ECR (container images) |
| **Cloud — Security** | AWS Secrets Manager · IAM instance roles · GitHub OIDC ephemeral credentials |
| **Observability** | Prometheus 2.54.1 · Sentry · Structured JSON logging · PSI + KS drift detection |
| **CI/CD** | GitHub Actions · Docker multi-stage builds · Ruff linter · pytest-asyncio |

---

## Dataset

**Twitter US Airline Sentiment** — [CrowdFlower / Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

14,640 annotated tweets across six U.S. carriers collected over nine days in February 2015. The dataset presents two structural challenges that make it a rigorous benchmark: significant class imbalance (62.7% negative), addressed through weighted cross-entropy loss during training; and a high prevalence of sarcastic complaint language — the primary driver motivating the DistilBERT architecture selection over its predecessors.

| Property | Value |
|---|---|
| Total tweets (post-deduplication) | ~14,427 |
| Airlines | United · US Airways · American · Southwest · Delta · Virgin America |
| Collection period | 16–24 February 2015 |
| Negative (Label 0) | 9,178 tweets — 62.7% |
| Neutral (Label 1) | 3,099 tweets — 21.2% |
| Positive (Label 2) | 2,363 tweets — 16.1% |
| Train / Validation / Test | 70% / 10% / 20% — stratified by class label |

---

## Term 2 Notebooks

The following notebooks constitute the academic comparative study from which the production model selection was derived. They are preserved as reference artefacts and are cited in the Term 3 project report.

| Notebook | Link | Scope |
|---|---|---|
| TF-IDF + Logistic Regression | [Open in Colab](https://colab.research.google.com/drive/1cghiSgwGtIjLEiPGdkqnJs7bQXOGBvo4) | Baseline model · 5-fold cross-validation · feature importance · issue-category analysis |
| DistilBERT Fine-Tuning | [Open in Colab](https://colab.research.google.com/drive/1fELjsGkZlpLW3eRIRCkUObyRLoso0xBq) | WeightedTrainer · 3 epochs · validation accuracy 84.99% |
| Bidirectional LSTM | [Open in Colab](https://colab.research.google.com/drive/1X_13ssDA1ghSDZVKBZt_MyScxu-yp6-M) | BiLSTM · test accuracy 75.61% · weighted F1 0.7663 |

---

## CLO Mapping — ADS 5711

| CLO | PLG | Learning Outcome | Project Demonstration |
|---|---|---|---|
| **CLO 1** | PLG 2 | Apply Strategic AI/ML Cloud Concepts | Three-phase comparative model study · evidence-based DistilBERT production selection · SageMaker as cloud ML platform · fully documented architectural decision rationale |
| **CLO 2** | PLG 3 | Integrate Cloud Infrastructure & MLOps | Three-workflow GitHub Actions CI/CD · AWS OIDC zero-static-key authentication · Docker multi-stage builds · ECR → ECS rolling deployments · SageMaker training job + managed endpoint · Redis + Celery background task infrastructure · AWS Secrets Manager runtime credential injection |
| **CLO 3** | PLG 4 | Manage, Monitor & Scale Cloud ML Workflows | PSI + KS statistical drift detection · daily Celery Beat scheduling · DriftRecord persistence with alert flags · Prometheus metrics · hourly MetricSnapshot aggregation · multi-tenant SaaS data model · on-demand retraining workflow via GitHub Actions |

---

## References

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
4. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.
5. Wolf, T., et al. (2020). *HuggingFace's Transformers: State-of-the-Art Natural Language Processing.* arXiv:1910.03771.
6. Amazon Web Services. (2024). *Amazon SageMaker Developer Guide.* docs.aws.amazon.com/sagemaker/
7. CrowdFlower / Figure Eight. (2015). *Twitter US Airline Sentiment Dataset.* Kaggle.

---

<div align="center">

MIT License

**Harsh Mimani** — [github.com/mimaniharsh](https://github.com/mimaniharsh) &nbsp;·&nbsp; **Gokul Raj E**

TAPMI MBA–AI&DS · Batch 2025–2027 · CRP ADS 5711 · Term 3

</div>
