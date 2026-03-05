<div align="center">

<br />

# Sentient Intelligence Platform

**Transform every customer signal into a strategic decision.**

Sentient is a production-grade ML platform that classifies sentiment, detects urgency, quantifies business impact, and surfaces root-cause clusters — across every channel, at enterprise scale.

<br />

[![CI](https://img.shields.io/github/actions/workflow/status/mimaniharsh/customer-sentiment-analysis-mlops/01-ci.yml?label=CI&style=flat-square&logo=github)](https://github.com/mimaniharsh/customer-sentiment-analysis-mlops/actions)
[![Model](https://img.shields.io/badge/Model-DistilBERT-4f46e5?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/distilbert-base-uncased)
[![AWS](https://img.shields.io/badge/Infra-SageMaker%20%7C%20S3%20%7C%20ECR-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-000000?style=flat-square&logo=nextdotjs)](https://nextjs.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br />

[**Live Demo**](#-demo-guide) · [**API Docs**](http://localhost:8000/docs) · [**Architecture**](#architecture) · [**Quick Start**](#quick-start)

<br />

</div>

---

## What Sentient Does

Most companies drown in unstructured feedback. Sentient turns the noise into a clear signal.

Every customer message — a tweet, a support ticket, an app review, a chat log — passes through a fine-tuned DistilBERT pipeline that produces far more than a sentiment label. It produces an *intelligence brief*: how negative, how urgent, what business area is affected, how confident the model is, which tokens drove the decision, and what it means for the bottom line.

The output flows into an executive dashboard where leaders see trends, clusters, and simulated impact — not raw data.

<br />

## The Intelligence Pipeline

Each message passes through six sequential analysis layers:

```
  Input Text
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  1. Sentiment Classification                             │
│     DistilBERT → Positive / Neutral / Negative          │
│     + calibrated confidence via temperature scaling      │
├─────────────────────────────────────────────────────────┤
│  2. Multi-Label Attribute Tagging                        │
│     Pricing · Support · Reliability · Onboarding        │
│     Performance · Features · Security                    │
├─────────────────────────────────────────────────────────┤
│  3. Urgency Detection                                    │
│     Critical / High / Medium / Low                       │
│     Composite: keyword signal + probability mass         │
├─────────────────────────────────────────────────────────┤
│  4. Severity & Business Impact Scoring                   │
│     Continuous 0.00–1.00 score                          │
│     Amplified by high-risk attribute co-occurrence       │
├─────────────────────────────────────────────────────────┤
│  5. Token-Level Explainability                           │
│     Attention-proxy saliency                             │
│     Top 5 driver tokens per prediction                   │
├─────────────────────────────────────────────────────────┤
│  6. Embedding Fingerprint                                │
│     For cluster membership lookup                        │
└─────────────────────────────────────────────────────────┘
      │
      ▼
  Structured JSON  →  PostgreSQL  →  Dashboard
```

All six layers resolve in a single API call, typically under **300ms**.

<br />

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Sentient Platform — AWS ap-south-1                │
│                                                                          │
│  ┌─────────────────┐   REST    ┌──────────────────┐  invoke   ┌───────┐ │
│  │  Next.js 15     │ ────────▶ │  FastAPI          │ ────────▶ │  SM   │ │
│  │  Dashboard      │           │  (multi-tenant,   │           │  End  │ │
│  │  TypeScript     │ ◀──────── │  RBAC, JWT)       │           │  point│ │
│  └─────────────────┘  JSON     └──────────────────┘           └───────┘ │
│                                        │                                 │
│                          ┌─────────────┴────────────┐                   │
│                          ▼                          ▼                   │
│                  ┌──────────────┐        ┌─────────────────────┐        │
│                  │  PostgreSQL  │        │  Celery + Redis      │        │
│                  │  Row-level   │        │  Drift detection     │        │
│                  │  tenant ISO  │        │  Metric snapshots    │        │
│                  └──────────────┘        │  Report generation   │        │
│                                          └─────────────────────┘        │
│                                                                          │
│  CI/CD    GitHub Actions → ECR → ECS   (OIDC · zero static keys)        │
│  ML Ops   S3 → SageMaker Training (GPU) → Model Registry → Endpoint     │
│  Secrets  AWS Secrets Manager (prod) · .env (dev)                       │
└──────────────────────────────────────────────────────────────────────────┘
```

<br />

## Capabilities

| Layer | Capability |
|---|---|
| **ML** | Fine-tuned DistilBERT · Calibrated confidence · Temperature scaling |
| **Classification** | 3-class sentiment · 7-attribute multi-label tagging |
| **Signals** | Urgency (4 levels) · Severity score · Business impact score |
| **Explainability** | Token-level saliency · Top-5 driver tokens per prediction |
| **Drift** | PSI + KS statistic · Rolling 12-week distribution monitoring |
| **Clustering** | Embedding-based root-cause topic grouping |
| **Platform** | Multi-tenant · RBAC (Owner / Admin / Analyst / Viewer) · Rate limiting |
| **Ops** | Celery workers · Redis queues · Prometheus metrics · Sentry tracing |

<br />

---

## 🎬 Demo Guide

The fastest way to see Sentient in action. The demo script feeds real customer feedback through the running platform and prints a colour-coded intelligence brief for every message in real time.

### Prerequisites

```bash
# Platform must be running
docker compose up

# One dependency for the demo script
pip install requests
```

---

### Mode 1 — Live Stream *(best for stakeholder demos)*

Streams messages one at a time with a configurable delay. Each message renders as a colour-coded intelligence card in the terminal, while the dashboard at **http://localhost:3000** updates live.

```bash
# Works immediately — no CSV required
python scripts/demo_live_stream.py

# Stream your Tweets.csv at a natural pace
python scripts/demo_live_stream.py --file Tweets.csv --delay 1.0

# Cinematic mode — 2 second pause between messages
python scripts/demo_live_stream.py --file Tweets.csv --delay 2.0 --limit 50 --shuffle

# Custom CSV with a non-standard text column
python scripts/demo_live_stream.py --file reviews.csv --text-col "review_text"
```

**Terminal output for each message:**

```
┌─────────────────────────────────────────────────────────────────┐
│ #0042  Third delay this week. No explanation, no apology.       │
│                                                                  │
│  Sentiment  NEGATIVE    Confidence 94.3%                        │
│  Urgency    HIGH        Impact     0.81/1.00                    │
│  Attributes reliability, support                                 │
│  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  pos 4%  neu 2%  neg 94%                            187ms       │
└─────────────────────────────────────────────────────────────────┘
```

---

### Mode 2 — Batch Ingest *(fastest — populate the dashboard in seconds)*

Sends messages in configurable batches. No delay. Processes thousands of rows per minute. Use this to seed the analytics dashboard before a live demo.

```bash
python scripts/demo_live_stream.py --mode batch --file Tweets.csv
python scripts/demo_live_stream.py --mode batch --file Tweets.csv --batch 50
```

---

### Mode 3 — Full CSV Upload *(structured report output)*

Uploads the entire file to the `/predict/csv` endpoint. Returns a structured JSON report, highlights the most critical messages, and saves results to `demo_results.json`.

```bash
python scripts/demo_live_stream.py --mode csv-upload --file Tweets.csv
```

```
✓ Upload complete — 14,640 rows analysed in 48.2s

⚠ 312 CRITICAL messages found:

  ● [0.94] Lost my luggage again. This is the second time in a month...
  ● [0.91] Customer support hung up on me after 45 minutes on hold...
  ● [0.89] Cancelled my flight with zero notice and no hotel voucher...

Summary:
  Positive:  5,821  (39.8%)
  Neutral:   3,104  (21.2%)
  Negative:  5,715  (39.0%)
  Critical:    312   (2.1%)
  Avg Confidence: 87.4%
```

---

### Mode 4 — Direct API *(curl / Postman / code)*

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Three hour delay and nobody at the gate to help us."}'

# Batch — up to 500 texts in one call
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great flight!", "Terrible service", "It was fine"]}'

# Upload a CSV file
curl -X POST http://localhost:8000/api/v1/predict/csv \
  -F "file=@Tweets.csv"
```

---

### Dashboard Pages

| Page | URL | What you see |
|---|---|---|
| Overview | http://localhost:3000 | KPI cards · health score · trend direction |
| Analytics | http://localhost:3000/analytics | Sentiment timeseries · attribute heatmap |
| Clusters | http://localhost:3000/clusters | Root-cause topic groupings |
| Reports | http://localhost:3000/reports | Exportable executive summaries |
| API Docs | http://localhost:8000/docs | Full interactive Swagger UI |

<br />

---

## Quick Start

### 1 — Clone

```bash
git clone https://github.com/mimaniharsh/customer-sentiment-analysis-mlops.git
cd customer-sentiment-analysis-mlops
```

### 2 — Configure

```bash
cp .env.example .env
```

Minimum required values in `.env`:

```env
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=ap-south-1
SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
S3_BUCKET=your-bucket-name
APP_SECRET_KEY=any-32-character-string
JWT_SECRET_KEY=any-other-32-character-string
```

### 3 — Start

```bash
docker compose up --build
```

First build takes ~4 minutes. Subsequent starts take ~20 seconds.

Six services start: PostgreSQL · Redis · FastAPI · Celery Worker · Celery Beat · Next.js

### 4 — Run the Demo

```bash
pip install requests
python scripts/demo_live_stream.py
```

Open http://localhost:3000. The dashboard updates in real time as messages are processed.

<br />

---

## ML Training Pipeline

Train a new model via GitHub Actions — no local GPU required.

**1.** Go to **Actions → ML — Train & Deploy SageMaker Model**  
**2.** Click **Run workflow** and set epochs + batch size  
**3.** The pipeline runs four stages automatically:

```
Data Preprocessing   (~5 min)   Kaggle → clean → split → S3
SageMaker Training   (~35 min)  ml.g4dn.xlarge GPU · DistilBERT fine-tune
Model Evaluation     (~5 min)   F1 · accuracy · confusion matrix
Endpoint Deployment  (~10 min)  Package → create/update SageMaker endpoint
```

### Required Secrets & Variables

| GitHub Secret | Description |
|---|---|
| `AWS_ROLE_ARN` | IAM role for GitHub OIDC (replaces static keys) |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role |
| `KAGGLE_USERNAME` | Kaggle account |
| `KAGGLE_KEY` | Kaggle API key |

| GitHub Variable | Example |
|---|---|
| `AWS_REGION` | `ap-south-1` |
| `S3_BUCKET` | `your-bucket` |
| `PROJECT_NAME` | `customer-sentiment-analysis-mlops` |
| `ENDPOINT_NAME` | `customer-sentiment-analysis-mlops-endpoint` |

<br />

---

## API Reference

Base path: `/api/v1/` · Full docs: http://localhost:8000/docs

### Prediction Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Single text — full 6-layer analysis |
| `POST` | `/predict/batch` | Up to 500 texts in one call |
| `POST` | `/predict/csv` | Upload CSV · up to 10,000 rows |

### Analytics Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/analytics/overview` | Executive KPI summary |
| `GET` | `/analytics/timeseries` | 30-day sentiment trend |
| `GET` | `/analytics/attributes` | Attribute heatmap data |
| `GET` | `/analytics/urgency` | Urgency distribution |
| `GET` | `/analytics/drift` | PSI / KS drift history |
| `GET` | `/analytics/clusters` | Root-cause cluster insights |
| `GET` | `/analytics/business-impact` | Scenario simulation |

### Example Response

```json
{
  "sentiment": "negative",
  "sentiment_score": -0.8841,
  "confidence": 0.9431,
  "probabilities": {
    "positive": 0.0281,
    "neutral": 0.0288,
    "negative": 0.9431
  },
  "urgency": "high",
  "urgency_score": 0.7821,
  "severity_score": 0.8743,
  "business_impact_score": 0.8156,
  "attributes": ["support", "reliability"],
  "attribute_scores": { "support": 0.67, "reliability": 0.33 },
  "explanation": {
    "sentiment_label": "negative",
    "top_tokens": [
      { "token": "terrible", "importance": 0.3214 },
      { "token": "unhelpful", "importance": 0.2891 }
    ],
    "method": "attention_proxy"
  },
  "latency_ms": 187
}
```

<br />

---

## Repository Structure

```
sentient-platform/
├── apps/
│   ├── api/                         FastAPI backend
│   │   └── app/
│   │       ├── core/config.py       Runtime secret loading
│   │       ├── api/v1/endpoints/    predict · analytics · auth · reports
│   │       ├── ml/pipeline.py       6-layer NLP enrichment engine
│   │       ├── models/db.py         SQLAlchemy async · tenant isolation
│   │       └── workers/             Celery tasks · beat scheduler
│   └── web/                         Next.js 15 dashboard
│       └── src/
│           ├── app/                 App Router pages
│           ├── components/          KPI cards · charts · simulators
│           └── lib/api.ts           Axios client · auth interceptors
│
├── packages/ml/
│   ├── pipeline/                    data_download · data_preprocess
│   ├── training/model_train.py      Weighted DistilBERT trainer
│   ├── evaluation/model_eval.py     Test-set metrics
│   └── serving/model_deploy.py      Endpoint deployment
│
├── scripts/
│   └── demo_live_stream.py          ← Start here for demos
│
├── .github/workflows/
│   ├── 01-ci.yml                    Lint · test · scan · Docker build
│   ├── 02-deploy.yml                ECS deploy via OIDC
│   └── 03-ml-train-deploy.yml       Full SageMaker pipeline
│
├── docker-compose.yml               Full local stack
└── .env.example                     Configuration template
```

<br />

---

## Secret Management

Zero secrets in the repository. Enforced by Gitleaks on every CI run.

| Environment | Source | Mechanism |
|---|---|---|
| Local | `.env` file | pydantic-settings at process start |
| Production | AWS Secrets Manager | Fetched at startup via IAM role |
| CI/CD | GitHub Actions + OIDC | Role assumed — no long-lived keys stored |

<br />

---

## Observability

| Signal | Implementation |
|---|---|
| Metrics | Prometheus · `/metrics` · 15s scrape |
| Logging | Structured JSON · automatic credential redaction |
| Errors | Sentry · environment-tagged |
| Health | `/health` liveness · `/api/v1/health` deep check |
| Latency | `X-Process-Time-Ms` response header |

<br />

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Training | PyTorch · HuggingFace Transformers · DistilBERT |
| ML Serving | AWS SageMaker managed endpoints |
| Backend | FastAPI · SQLAlchemy async · Alembic · Celery |
| Database | PostgreSQL 16 |
| Cache / Queue | Redis 7 |
| Frontend | Next.js 15 · React 19 · TypeScript · Tailwind CSS |
| Visualisation | Recharts |
| Auth | JWT · bcrypt |
| Cloud | AWS — SageMaker · S3 · ECR · ECS · Secrets Manager |
| CI/CD | GitHub Actions · Docker · AWS OIDC |

<br />

---

## Cost Reference

| Resource | Approximate Cost |
|---|---|
| `ml.g4dn.xlarge` training job | ~$0.70–1.00/hour (billed per second) |
| `ml.m5.large` inference endpoint | ~$0.115/hour (~$83/month) |
| `docker compose up` locally | $0 |

Delete the endpoint between sessions to avoid idle charges:

```bash
aws sagemaker delete-endpoint \
  --endpoint-name customer-sentiment-analysis-mlops-endpoint \
  --region ap-south-1
```

Redeploy anytime via **Actions → ML — Train & Deploy SageMaker Model**.

<br />

---

<div align="center">

MIT License · © Sentient Intelligence Platform

*Built with precision. Deployed with confidence.*

</div>
