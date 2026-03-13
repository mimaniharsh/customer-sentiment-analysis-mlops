<div align="center">

<img src="https://img.shields.io/badge/Sentient-Intelligence%20Platform-4f46e5?style=for-the-badge&logoColor=white" alt="Sentient" height="32"/>

# Customer Sentiment Analysis & Brand Reputation Tracking

**A production-grade NLP platform that goes beyond labels — delivering sarcasm detection, severity triage, root-cause attribution, and real-time escalation intelligence.**

<br/>

[![CI](https://img.shields.io/github/actions/workflow/status/mimaniharsh/customer-sentiment-analysis-mlops/01-ci.yml?label=CI%20Pipeline&style=flat-square&logo=github)](https://github.com/mimaniharsh/customer-sentiment-analysis-mlops/actions)
[![Model](https://img.shields.io/badge/Models-TF--IDF%20·%20BiLSTM%20·%20DistilBERT-4f46e5?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/distilbert-base-uncased)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20·%20ECS%20·%20S3-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2015-000000?style=flat-square&logo=nextdotjs)](https://nextjs.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Dataset](https://img.shields.io/badge/Dataset-Twitter%20Airline%20Sentiment-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

<br/>

[**Quick Start**](#-quick-start) · [**Model Comparison**](#-model-comparison) · [**API Reference**](#-api-reference) · [**Architecture**](#-architecture) · [**Dashboard**](#-dashboard) · [**ML Pipeline**](#-ml-training-pipeline)

<br/>

> **Academic Context** — TAPMI MBA-AI&DS · Concurrent Running Project (CRP) · ADS 5711 · Term 3
> Submitted by **Harsh Mimani** (25A108) & **Gokul Raj E** (25A106)

</div>

---

## The Problem

Airlines receive thousands of tweets every day. Standard keyword-based sentiment tools fail systematically on the most important signal: **sarcasm**.

> *"Oh brilliant, delayed again. Truly phenomenal service."*

Every keyword system in the industry scores that tweet as **positive**. The customer is furious and about to churn.

**Sentient** solves this. It evaluates three NLP architectures side-by-side on the same dataset, deploys the best-performing model to production, and runs every prediction through a four-layer enrichment engine that returns not just a sentiment label — but an **actionable intelligence card** complete with urgency level, severity tier, sarcasm probability, root-cause attribution, business impact score, and a specific escalation action.

---

## What This Delivers

| Without Sentient | With Sentient |
|---|---|
| Manual tweet triage: 3–5 min/tweet | Automated triage: < 200 ms/tweet |
| Sarcasm misclassified as positive | 82%+ sarcasm detection with override & explanation |
| Human judgment on escalation priority | Deterministic 4-tier severity matrix, auditable criteria |
| Neutral class F1 of 0.44 (TF-IDF) | Neutral class F1 of 0.82 (DistilBERT) |
| Ad hoc root-cause labelling | Automated 7-category × 35-issue decomposition per tweet |
| No model quality monitoring | Daily PSI + KS drift detection with alert thresholds |

---

## Model Comparison

Three models. One dataset. One production winner — chosen on evidence, not assumption.

| Model | Approach | F1 (Weighted) | Test Accuracy | Neutral F1 | Inference | Best For |
|---|---|:---:|:---:|:---:|:---:|---|
| **TF-IDF + LR** | Bag-of-words · linear classifier | ~0.79 | ~72% | 0.44 | < 5 ms | Compliance audits, explainability |
| **BiLSTM** | Sequence model · trainable embeddings | ~0.84 | ~76% | 0.60 | ~15 ms | Streaming, sequential context |
| **DistilBERT** ✦ | Fine-tuned transformer · 66M params | **~0.91** | **~91%** | **0.82** | ~180 ms | **Production API** — handles sarcasm, negation, ambiguity |

> ✦ **Production model.** Metrics are logged to S3 after every SageMaker training run and surface in the GitHub Actions workflow summary.

### Why DistilBERT Wins in Production

Bidirectional context encoding handles sarcasm, negation, and semantic ambiguity — the three failure modes where both TF-IDF and LSTM fall short on this dataset. At 66M parameters versus BERT-base's 110M, it delivers 97% of BERT's accuracy at 60% of the inference cost.

### Why TF-IDF Stays in the Stack

When a classification must be explained to a non-technical stakeholder, a regulator, or a compliance team, coefficient weights are unambiguous. The comparative architecture keeps the baseline available as an auditable, always-interpretable fallback.

---

## The Intelligence Pipeline

Every prediction clears the base model, then passes through four enrichment heads before a response is returned.

```
  Input Tweet
       │
       ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Model Layer                                                            │
 │  Configurable: TF-IDF + LR · BiLSTM · DistilBERT (default)              │
 │  Output: Positive / Neutral / Negative + raw class probabilities        │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
              ┌─────────────────┼───────────────────┐
              ▼                 ▼                   ▼
    ┌──────────────────┐ ┌──────────────┐ ┌────────────────────┐
    │  Sarcasm Head    │ │ Severity Head│ │  Attribute Tagger  │
    │                  │ │              │ │                    │
    │  4 independent   │ │  4-tier      │ │  7 business labels │
    │  signals:        │ │  keyword     │ │  pricing · support │
    │  · trigger+ctx   │ │  matrix      │ │  reliability ·     │
    │  · prob mismatch │ │  CRITICAL    │ │  performance ·     │
    │  · negation flip │ │  HIGH        │ │  onboarding ·      │
    │  · exclamation   │ │  MEDIUM      │ │  features ·        │
    │  amplifier       │ │  LOW         │ │  security          │
    └────────┬─────────┘ └──────┬───────┘ └────────┬───────────┘
             │                  │                   │
             └──────────────────┼───────────────────┘
                                ▼
                  ┌──────────────────────────────┐
                  │  Override + Risk Tier Engine │
                  │  Sarcasm ≥ 55% → label flip  │
                  │  Severity CRITICAL → flip    │
                  │  Override reason always      │
                  │  included in response        │
                  └─────────────┬────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────┐
             │         Intelligence Card            │
             │  sentiment · confidence · sarcasm    │
             │  severity · urgency · impact score   │
             │  attributes · root-cause breakdown   │
             │  strategic insight · top tokens      │
             │  escalation action · latency_ms      │
             └──────────────────────────────────────┘
```

### A Live Example

```
Input:   "Oh wow, another delay. Just what I needed on a Monday. Phenomenal service."

Raw DistilBERT output:
  → Positive 68.1%  ·  Neutral 4.0%  ·  Negative 27.9%

Enrichment pipeline:
  Sarcasm      82%    Trigger 'phenomenal' + context 'delay' (0.35)
                      Probability mismatch — positive label, prob_neg 27.9% (0.14)
                      Exclamation on complaint topic (0.15)

  OVERRIDE     positive → NEGATIVE
               Sarcasm score 82% exceeds threshold of 55%

  Severity     HIGH  ·  Keywords: ['delay']  ·  Criteria: service failure
  Risk         HIGH  ·  Assign dedicated team within 4 hours
  Attributes   reliability, performance
  Impact       0.61 / 1.00
  Insight      "High-severity reliability failure detected. Unresolved signals
                in this tier correlate with elevated churn risk."
```

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, an AWS account with SageMaker access.

```bash
# 1. Clone the repository
git clone https://github.com/mimaniharsh/customer-sentiment-analysis-mlops.git
cd customer-sentiment-analysis-mlops

# 2. Configure environment
cp .env.example .env
# Open .env and fill in your values (see below)

# 3. Start all six services
docker compose up --build
# First build: ~4 min  ·  Subsequent starts: ~20s
```

### Minimum `.env` Configuration

```env
# AWS
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=ap-south-1
SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
S3_BUCKET=your-s3-bucket-name

# Application
APP_SECRET_KEY=any-32-char-string
JWT_SECRET_KEY=another-32-char-string
```

### Access Points

| URL | Service |
|---|---|
| http://localhost:3000 | Executive dashboard (Next.js) |
| http://localhost:8000/docs | Interactive API — Swagger UI |
| http://localhost:9090 | Prometheus metrics |

### Run the Demo

```bash
# Install the demo dependency
pip install requests

# Stream tweets through the pipeline in real time
python scripts/demo_live_stream.py

# Batch-process an entire CSV
python scripts/demo_live_stream.py --mode batch --file Tweets.csv

# Presentation mode — cinematic 2-second pacing
python scripts/demo_live_stream.py --delay 2.0 --limit 50 --shuffle
```

---

## API Reference

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Three delays this week. Absolute shambles."}'
```

### Batch Prediction — up to 500 texts per call

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great flight!", "Total disaster.", "It was fine."]}'
```

### CSV Upload — up to 10,000 rows

```bash
curl -X POST http://localhost:8000/api/v1/predict/csv \
  -F "file=@Tweets.csv"
```

### Full Response Shape

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
  "sarcasm_signals": ["Trigger 'phenomenal' with context: {'delay'}"],
  "sentiment_override_reason": "Overridden from 'positive': sarcasm 82% exceeds threshold.",
  "severity_tier": "HIGH",
  "severity_score": 0.61,
  "severity_criteria": ["Significant service failure — escalation warranted"],
  "urgency": "high",
  "risk": {
    "tier": "HIGH",
    "action": "Assign dedicated team within 4 hours.",
    "label": "Service recovery required"
  },
  "attributes": ["reliability", "performance"],
  "theme_breakdown": [
    {
      "clause": "another delay",
      "attribute": "reliability",
      "issue": "repeated_delay",
      "mini_sentiment": "negative"
    }
  ],
  "business_impact_score": 0.61,
  "strategic_insight": "High-severity reliability failure detected. Unresolved signals in this tier correlate with elevated churn risk.",
  "explanation": {
    "top_tokens": [
      { "token": "shambles", "importance": 0.34 },
      { "token": "delays", "importance": 0.29 }
    ]
  },
  "latency_ms": 187
}
```

### Severity Escalation Matrix

| Tier | Score Range | Trigger Keywords | Escalation Action |
|---|:---:|---|---|
| **CRITICAL** | 0.75 – 1.00 | cancel, refund, fraud, stranded, never again, data breach | Senior team · within 1 hour · executive review |
| **HIGH** | 0.50 – 0.75 | delay, broken, terrible, unresolved, failure, disaster | Dedicated team · within 4 hours · weekly flag |
| **MEDIUM** | 0.25 – 0.50 | slow, confusing, disappointing, mediocre, improve | Response within 48 hours · sprint inclusion |
| **LOW** | 0.00 – 0.25 | *(no negative signals)* | Log for trend · monthly report |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Sentient Platform — AWS                            │
│                                                                        │
│   ┌───────────────┐      ┌──────────────────────────┐                  │
│   │  Next.js 15   │      │   FastAPI (async)        │                  │
│   │  Dashboard    │────▶│   Multi-tenant REST API  │                  │
│   │  React 19     │◀────│   JWT + RBAC             │                  │
│   │  TypeScript   │      │   SlowAPI rate limiting  │                  │
│   └───────────────┘      └───────────┬──────────────┘                  │
│                                      │                                 │
│                         ┌────────────▼────────────┐                    │
│                         │  SageMaker Endpoint     │                    │
│                         │  DistilBERT inference   │                    │
│                         │  ~180ms · ml.m5.large   │                    │
│                         └─────────────────────────┘                    │
│                                                                        │
│   ┌─────────────────────┐       ┌───────────────────────────────┐      │
│   │   PostgreSQL 16     │       │  Redis 7 + Celery             │      │
│   │   7 async models    │       │  Worker (concurrency=4)       │      │
│   │   Row-level tenant  │       │  Beat scheduler               │      │
│   │   isolation         │       │  · Drift detection (02:00 UTC)│      │
│   └─────────────────────┘       │  · Hourly metric snapshots    │      │
│                                 │  · Async report generation    │      │
│                                 └───────────────────────────────┘      │
│                                                                        │
│   CI/CD     GitHub Actions (OIDC) → ECR → ECS rolling deploy           │
│   Secrets   AWS Secrets Manager (prod) · .env (dev)                    │
│   Monitor   Prometheus · Sentry · Structured JSON logs · PSI drift     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Service Inventory

| Service | Technology | Port | Role |
|---|---|:---:|---|
| `api` | FastAPI 0.115 + Uvicorn (uvloop) | 8000 | Core ML API — predictions, analytics, auth, reports |
| `worker` | Celery 5.4 (concurrency=4) | — | Drift detection, metric snapshots, async reports |
| `beat` | Celery Beat | — | Cron scheduler — hourly metrics, daily drift at 02:00 UTC |
| `web` | Next.js 15 + React 19 + TypeScript | 3000 | Executive dashboard — 7 pages, real-time charts |
| `postgres` | PostgreSQL 16 Alpine | 5432 | Primary store — 7 tables, row-level tenant isolation |
| `redis` | Redis 7 Alpine | 6379 | Celery broker (DB 1) + API response cache (DB 0) |
| `prometheus` | Prometheus 2.54 | 9090 | Metrics scraping — API latency, endpoint health |

---

## Repository Structure

```
customer-sentiment-analysis-mlops/
│
├── apps/
│   ├── api/
│   │   └── app/
│   │       ├── api/v1/endpoints/        # predict · analytics · reports · auth · health
│   │       ├── ml/
│   │       │   └── pipeline.py          # 4-head enrichment engine (920+ lines)
│   │       ├── models/db.py             # 7 async SQLAlchemy models — multi-tenant schema
│   │       ├── core/                    # config · database · security · deps
│   │       └── workers/                 # Celery — drift · metrics · report generation
│   │
│   └── web/src/app/
│       ├── page.tsx                     # Overview — health score, NPS trend, alerts
│       ├── analytics/                   # Sentiment timeseries, attribute heatmap
│       ├── clusters/                    # Root-cause topic grouping, negative rates
│       ├── demo/                        # Live prediction stream, IntelligenceCard
│       ├── reports/                     # 1-click report generation suite
│       ├── strategy/                    # Business impact simulator
│       └── settings/                    # Tenant config, API key management
│
├── packages/ml/
│   ├── pipeline/
│   │   ├── data_download.py             # Kaggle download via kagglehub
│   │   └── data_preprocess.py           # Cleaning pipeline + stratified splits
│   ├── training/
│   │   └── model_train.py               # DistilBERT WeightedTrainer — 10 engineering fixes
│   ├── evaluation/
│   │   └── model_eval.py                # Per-class F1, confusion matrix, ECE calibration
│   └── serving/
│       └── model_deploy.py              # SageMaker endpoint create/update
│
├── .github/workflows/
│   ├── 01-ci.yml                        # Lint · test · secret scan · Docker build
│   ├── 02-deploy.yml                    # ECS rolling deploy via OIDC (manual trigger)
│   └── 03-ml-train-deploy.yml           # Full SageMaker pipeline (manual trigger)
│
├── scripts/
│   └── demo_live_stream.py              # ← Start here for a live demo
│
├── infra/docker/
│   ├── nginx.conf
│   └── prometheus.yml
│
├── docker-compose.yml                   # 6-service local dev stack — one command
├── .env.example                         # Credential template — never commit .env
└── SETUP.md                             # Full deployment guide
```

---

## ML Training Pipeline

The model training runs entirely on AWS — no local GPU required. Triggered manually from GitHub Actions.

### Training Workflow — 4 Sequential Jobs

| Stage | Duration | Infrastructure | What Happens |
|---|:---:|---|---|
| **Preprocess** | ~5 min | GitHub runner (Ubuntu) | Download dataset via Kaggle API → clean → stratified 70/10/20 split → upload CSVs to S3 |
| **Train** | ~35 min | SageMaker `ml.g4dn.xlarge` (NVIDIA T4 GPU) | DistilBERT fine-tuning — early stopping, weighted cross-entropy, cosine LR schedule |
| **Evaluate** | ~5 min | GitHub runner | Batched inference on test set → per-class F1, confusion matrix, ECE calibration → S3 + GitHub Actions artifacts |
| **Deploy** | ~10 min | GitHub runner + SageMaker | Package model with production `inference.py` → create/update managed endpoint |

**Trigger:** GitHub Actions → `ML — Train & Deploy SageMaker Model` → Run workflow (choose epochs and batch size)

### Key Training Decisions

```
Weighted CrossEntropy   →  Negative tweets = 63% of dataset. Without class weighting,
                           the model learns to ignore minority classes.

Label Smoothing 0.10    →  Prevents overconfident softmax. A displayed confidence of
                           82% should mean ~82% of such predictions are correct.

Cosine LR Schedule      →  Smoother final-epoch convergence vs linear decay.
                           More stable production behaviour across runs.

Early Stopping (p=3)    →  Halts training before overfitting. Saves ~$0.30–0.50
                           per run at current SageMaker pricing.

Temperature Scaling T=1.3 → Post-processes raw probabilities for calibration.
                             Expected Calibration Error (ECE) < 0.05 is excellent;
                             this is measured explicitly in the evaluation job.

Dynamic Padding         →  Per-batch padding (DataCollatorWithPadding) rather than
                           static dataset-wide padding — reduces wasted compute.
```

### Expected Model Performance

| Metric | Value |
|---|---|
| Validation Accuracy | ~91% |
| Weighted F1 | ~0.91 |
| F1 — Negative Class | ~0.93 |
| F1 — Neutral Class | ~0.82 |
| F1 — Positive Class | ~0.87 |
| Inference Latency | ~180ms (ml.m5.large endpoint) |
| Training Cost | ~$0.43 per run |
| Endpoint Cost | ~$83/month while active |

---

## Dashboard

Seven purpose-built pages, each designed for a specific stakeholder persona.

| Page | Route | Built For | Key Components |
|---|---|---|---|
| **Overview** | `/` | Executive / Board | Brand health score, 30-day NPS trend, critical alert count, drift indicator |
| **Analytics** | `/analytics` | Operations / CX | Sentiment timeseries, attribute heatmap (7 categories × time), urgency breakdown |
| **Clusters** | `/clusters` | Product / Engineering | Root-cause topic groups, ClusterInsights, negative rate per theme |
| **Live Demo** | `/demo` | Engineering / Sales | Real-time IntelligenceCard per tweet, bubble chart (severity × confidence) |
| **Reports** | `/reports` | Analysts / Management | 1-click executive summary, risk breakdown, drift analysis — async via Celery |
| **Strategy** | `/strategy` | Leadership | Business impact simulator — forecast NPS impact of resolving issue categories |
| **Settings** | `/settings` | Admin | Tenant config, API key management, model version selection |

---

## MLOps & CI/CD

### Three GitHub Actions Workflows

**`01-ci.yml`** — Runs on every push to `main`/`develop` and every pull request.
- Secret scanning: regex scan for hardcoded AWS key patterns (`AKIA[0-9A-Z]{16}`) across all Python, TypeScript, and YAML files. Any hit **fails the build immediately**.
- Python CI: Ruff linter + pytest with async httpx test client.
- Frontend CI: Next.js production build verification.

**`02-deploy.yml`** — Manual trigger only (staging or production environment).
- Authenticates to AWS via OIDC (`id-token: write`) — **zero static AWS keys stored in GitHub**.
- Builds and pushes API + Web Docker images to ECR (tagged by Git SHA for immutability).
- Deploys both services to ECS with rolling update and zero downtime.

**`03-ml-train-deploy.yml`** — Manual trigger with epoch and batch size inputs.
- Runs all four pipeline stages: preprocess → train → evaluate → deploy.
- All evaluation artifacts uploaded to both S3 and the GitHub Actions summary.

### Drift Detection

The `DriftDetector` class implements two complementary statistics:

- **Population Stability Index (PSI)** — compares recent sentiment score distributions against a reference window. PSI > 0.2 triggers a drift alert stored to the `drift_records` table.
- **Kolmogorov–Smirnov (KS) statistic** — measures distributional shift in probability outputs.

The Celery Beat scheduler runs drift detection daily at **02:00 UTC**. Alerts surface on the Overview dashboard and are flagged for human review.

---

## Security

Zero credentials anywhere in the codebase. The CI workflow actively enforces this — secret scanning runs as the first job and will fail the build before any other work proceeds.

| Environment | Secret Source | Mechanism |
|---|---|---|
| **Local development** | `.env` file (git-ignored) | pydantic-settings reads at app startup |
| **Production (AWS)** | AWS Secrets Manager | Fetched at container startup via IAM instance role — zero static keys |
| **CI/CD (GitHub Actions)** | GitHub Secrets + OIDC | Ephemeral role assumed per workflow run — no long-lived access keys |

**Additional security controls:** JWT authentication (python-jose + bcrypt), RBAC with four roles (owner / admin / analyst / viewer), input validation via Pydantic v2 field validators, SQL injection prevention via parameterised ORM queries, CORS with explicit allow-list in production, and log redaction filtering for any credential patterns.

---

## Cost Profile

| Resource | Cost |
|---|---|
| SageMaker training (`ml.g4dn.xlarge`) | ~$0.43 per run (~35 min) |
| SageMaker inference endpoint (`ml.m5.large`) | ~$83/month while active |
| ECS Fargate (API + Web) | ~$20–30/month |
| PostgreSQL on EC2 or RDS (`db.t3.micro`) | ~$15–25/month |
| S3 (models, data, reports) | < $5/month |
| **Total (endpoint active)** | **~$120–150/month** |
| **Total (endpoint deleted, redeploy on demand)** | **~$40–60/month** |

**Delete the endpoint between sessions to save ~$83/month:**

```bash
aws sagemaker delete-endpoint \
  --endpoint-name customer-sentiment-analysis-mlops-endpoint \
  --region ap-south-1

# Redeploy any time — one click via GitHub Actions workflow_dispatch
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **NLP Models** | TF-IDF + Logistic Regression · Bidirectional LSTM · DistilBERT (HuggingFace Transformers + PyTorch) |
| **ML Platform** | AWS SageMaker — training jobs, model registry, managed inference endpoints |
| **Backend** | FastAPI 0.115 · SQLAlchemy async · Alembic migrations · Celery 5.4 · pydantic v2 |
| **Data** | PostgreSQL 16 · Redis 7 |
| **Frontend** | Next.js 15 · React 19 · TypeScript · Tailwind CSS · Recharts |
| **Cloud** | AWS — SageMaker · S3 · ECR · ECS Fargate · Secrets Manager |
| **CI/CD** | GitHub Actions · Docker multi-stage builds · AWS OIDC (zero static keys) |
| **Observability** | Prometheus 2.54 · Sentry (distributed tracing) · Structured JSON logging · PSI + KS drift detection |

---

## Dataset

**Twitter US Airline Sentiment** — [CrowdFlower / Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

| Property | Value |
|---|---|
| Total tweets | 14,640 |
| Airlines | United · US Airways · American · Southwest · Delta · Virgin America |
| Collection period | 16–24 February 2015 |
| Negative | 9,178 tweets (62.7%) |
| Neutral | 3,099 tweets (21.2%) |
| Positive | 2,363 tweets (16.1%) |
| Train / Validation / Test split | 70% / 10% / 20% — stratified |

The class imbalance (63% negative) is a core challenge addressed by weighted cross-entropy during training and class-weight sampling during evaluation. The dataset's high prevalence of sarcasm in airline complaints makes it a rigorous stress test for all three model architectures.

---

## Project Notebooks (Term 2 Baseline)

These notebooks document the exploratory and comparative phase from which the production architecture was derived.

| Notebook | Link | Purpose |
|---|---|---|
| TF-IDF + Logistic Regression | [Open in Colab](https://colab.research.google.com/drive/1cghiSgwGtIjLEiPGdkqnJs7bQXOGBvo4) | Baseline model, 5-fold CV, feature importance, issue-category analysis |
| DistilBERT Fine-Tuning | [Open in Colab](https://colab.research.google.com/drive/1fELjsGkZlpLW3eRIRCkUObyRLoso0xBq) | WeightedTrainer, 3 epochs, validation accuracy 84.99% |
| BiLSTM | [Open in Colab](https://colab.research.google.com/drive/1X_13ssDA1ghSDZVKBZt_MyScxu-yp6-M) | Bidirectional LSTM, test accuracy 75.61%, weighted F1 0.7663 |

---

## References

1. Sanh, V., et al. (2020). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108
2. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805
3. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
4. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.
5. Wolf, T., et al. (2020). *HuggingFace's Transformers: State-of-the-Art NLP.* arXiv:1910.03771.
6. CrowdFlower / Figure Eight. (2015). *Twitter US Airline Sentiment Dataset.* Kaggle.

---

<div align="center">

MIT License

**Harsh Mimani** · [github.com/mimaniharsh](https://github.com/mimaniharsh) &nbsp;·&nbsp; **Gokul Raj E**

TAPMI MBA-AI&DS · Concurrent Running Project (CRP) · ADS 5711 · Term 3 · Batch 2025–2027

*Supervised by Prof. Winson Fernandes (MSIS) & Prof. Gaurav Sarin (TAPMI)*

</div>
