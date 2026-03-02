<div align="center">

# Sentient Intelligence Platform

**Enterprise Customer Sentiment & Branding Intelligence**

*Built for the scale of modern enterprise. Designed for the clarity of executive decision-making.*

[![CI](https://github.com/your-org/sentient-platform/actions/workflows/01-ci.yml/badge.svg)](https://github.com/your-org/sentient-platform/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-violet.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)

</div>

---

## What Is Sentient?

Sentient is a **production-grade, multi-tenant SaaS platform** that transforms raw customer feedback — tweets, support tickets, app reviews, chat logs, and long-form surveys — into executive-grade intelligence. Powered by a fine-tuned DistilBERT transformer pipeline on AWS SageMaker, Sentient delivers sentiment classification, urgency triage, severity scoring, business impact weighting, explainability, and proactive model drift alerting at enterprise scale.

Organizations running Sentient gain a single pane of glass across every customer signal: from individual message-level explanations to board-ready KPI dashboards, scenario simulators, and automated root-cause clustering — all in real time, all under full tenant isolation with role-based access control.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Sentient Intelligence Platform                   │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐│
│  │  Next.js 15  │───▶│  FastAPI Backend  │───▶│  SageMaker Endpoint   ││
│  │  Dashboard   │    │  (multi-tenant,   │    │  DistilBERT NLP       ││
│  │  (TypeScript)│◀───│   RBAC, JWT)      │    │  (ap-south-1)         ││
│  └──────────────┘    └──────────────────┘    └────────────────────────┘│
│                               │   │                                     │
│                    ┌──────────┘   └─────────┐                          │
│                    ▼                        ▼                           │
│           ┌──────────────┐       ┌────────────────────┐               │
│           │  PostgreSQL  │       │  Celery + Redis     │               │
│           │  (multi-     │       │  (drift detection,  │               │
│           │   tenant DB) │       │   metric snapshots, │               │
│           └──────────────┘       │   report generation)│               │
│                                  └────────────────────┘               │
│                                                                         │
│  Secret Management: AWS Secrets Manager (prod) · .env file (dev)       │
│  Observability:     Prometheus · Sentry · Structured Logs              │
│  CI/CD:             GitHub Actions · OIDC · ECR · ECS                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 🧠 ML Intelligence Pipeline
| Capability | Detail |
|---|---|
| Sentiment Classification | Positive · Neutral · Negative via fine-tuned DistilBERT |
| Multi-Label Attributes | Pricing · Support · Reliability · Onboarding · Features · Performance · Security |
| Urgency Detection | Critical / High / Medium / Low with composite keyword + probability scoring |
| Severity Scoring | 0–100 composite score from sentiment probability + urgency |
| Business Impact Weighting | Amplified by high-risk attribute co-occurrence |
| Calibrated Confidence | Temperature-scaled softmax probabilities |
| Explainability | Token-level saliency (attention proxy / Integrated Gradients ready) |
| Drift Detection | PSI + KS statistic on rolling 12-week sentiment distributions |
| Root-Cause Clustering | Embedding-based K-means topic grouping |
| Model Versioning | Every prediction stamped with model version + endpoint provenance |

### 📊 Dashboard & Analytics
- Executive KPI overview with 8-metric card grid
- 30-day sentiment trend area chart with area fills
- Urgency breakdown horizontal bar chart
- Attribute heatmap (7 attributes × 3 sentiments)
- Model drift line chart with threshold reference line
- Root-cause cluster cards with theme, count, severity, and top terms
- Business impact scenario simulator (interactive sliders)

### 🏢 Enterprise Platform
- **Multi-tenancy** — Full row-level tenant isolation on every DB table and API call
- **RBAC** — Owner / Admin / Analyst / Viewer roles with fine-grained endpoint protection
- **Rate Limiting** — Per-IP via SlowAPI, configurable per plan
- **Background Workers** — Celery + Redis for async metric snapshots, drift checks, and report generation
- **Exportable Reports** — S3-backed PDF/XLSX report generation pipeline

---

## Repository Structure

```
sentient-platform/
├── apps/
│   ├── api/                        # FastAPI backend
│   │   ├── app/
│   │   │   ├── core/
│   │   │   │   └── config.py       # Runtime secret loading (env / Secrets Manager)
│   │   │   ├── api/v1/endpoints/   # predict · analytics · auth · tenants · reports
│   │   │   ├── ml/
│   │   │   │   └── pipeline.py     # Full NLP enrichment pipeline
│   │   │   ├── models/db.py        # SQLAlchemy async models
│   │   │   ├── workers/            # Celery tasks + beat scheduler
│   │   │   └── main.py             # App factory with middleware stack
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── web/                        # Next.js 15 frontend
│       ├── src/
│       │   ├── app/                # App Router pages
│       │   ├── components/
│       │   │   ├── charts/         # Recharts visualization components
│       │   │   └── dashboard/      # KPI cards · Sidebar · TopBar · Simulators
│       │   └── lib/api.ts          # Axios client with auth interceptors
│       └── Dockerfile
│
├── packages/ml/                    # SageMaker ML pipeline scripts
│   ├── pipeline/                   # data_download.py · data_preprocess.py
│   ├── training/model_train.py     # Weighted DistilBERT trainer
│   ├── evaluation/model_eval.py    # Evaluation metrics
│   └── serving/model_deploy.py    # Endpoint deployment automation
│
├── infra/
│   ├── docker/prometheus.yml       # Prometheus scrape config
│   ├── terraform/                  # (extend with ECS, RDS, ElastiCache)
│   └── k8s/                        # (extend with Helm charts)
│
├── .github/workflows/
│   ├── 01-ci.yml                   # Lint · type-check · secret scan · Docker build
│   ├── 02-deploy.yml               # ECS deployment via OIDC (no static keys)
│   └── 03-ml-train-deploy.yml      # Full SageMaker train → evaluate → deploy
│
├── docker-compose.yml              # Full local dev stack (API · Web · PG · Redis · Prometheus)
├── .env.example                    # Template with placeholders only
├── .gitignore                      # Excludes .env, secrets, model weights, terraform state
└── README.md
```

---

## Secret Management Architecture

> **Zero secrets in the repository.** This is a hard architectural constraint, not a best-effort policy.

### How It Works

| Environment | Secret Source | Mechanism |
|---|---|---|
| Local Development | `.env` file on developer machine | `pydantic-settings` reads from `.env` at process start |
| Staging / Production | AWS Secrets Manager | `config.py` detects `APP_ENV=production` and fetches the secret JSON at startup via IAM role |
| CI/CD Pipelines | GitHub Actions Secrets + OIDC | No static keys — the deploy workflow assumes an IAM role via GitHub OIDC |

### Using Your Existing AWS Credentials

Your existing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are **not modified** by this platform. You plug them in at runtime only — they never touch the repository.

**Local development:**
```bash
cp .env.example .env
# Open .env and fill in:
# AWS_ACCESS_KEY_ID=<your-existing-key>
# AWS_SECRET_ACCESS_KEY=<your-existing-secret>
# AWS_REGION=ap-south-1
# SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
```

**Production (recommended):** Store the keys as a JSON object in AWS Secrets Manager:
```bash
aws secretsmanager create-secret \
  --name sentient/production/secrets \
  --secret-string '{
    "AWS_ACCESS_KEY_ID": "YOUR_KEY",
    "AWS_SECRET_ACCESS_KEY": "YOUR_SECRET",
    "APP_SECRET_KEY": "...",
    "JWT_SECRET_KEY": "...",
    "DATABASE_URL": "..."
  }' \
  --region ap-south-1
```
Then set `APP_ENV=production` and attach an IAM role with `secretsmanager:GetSecretValue` permission to your ECS task or EC2 instance. The API loads secrets automatically at startup with no keys in environment variables.

**Best path forward:** Migrate to an [IAM Role for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) which requires zero long-lived credentials at all.

---

## Quick Start

### Prerequisites
- Docker ≥ 24 and Docker Compose ≥ 2
- AWS credentials with SageMaker endpoint access (see above)
- Node.js 20+ (for frontend development only)
- Python 3.11+ (for local API development only)

### 1 — Clone and configure

```bash
git clone https://github.com/your-org/sentient-platform.git
cd sentient-platform

# Create your local secrets file (never committed)
cp .env.example .env
```

Edit `.env` with your values. **Minimum required:**
```env
AWS_ACCESS_KEY_ID=<your-existing-key>
AWS_SECRET_ACCESS_KEY=<your-existing-secret>
AWS_REGION=ap-south-1
SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
APP_SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
```

### 2 — Start the full stack

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API (FastAPI) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| PostgreSQL | localhost:5432 |

### 3 — Run your first prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The onboarding experience is incredibly smooth, loved it!"}'
```

**Response includes:**
```json
{
  "sentiment": "positive",
  "sentiment_score": 0.8421,
  "confidence": 0.9312,
  "urgency": "low",
  "urgency_score": 0.12,
  "severity_score": 0.08,
  "business_impact_score": 0.14,
  "attributes": ["onboarding"],
  "attribute_scores": { "onboarding": 0.67 },
  "explanation": {
    "sentiment_label": "positive",
    "top_tokens": [
      { "token": "smooth", "importance": 0.31 },
      { "token": "loved", "importance": 0.28 }
    ]
  }
}
```

---

## ML Pipeline — Train a New Model

The ML training pipeline is fully automated via GitHub Actions.

### Trigger a training run

1. Navigate to **Actions → ML — Train & Deploy SageMaker Model**
2. Click **Run workflow**, set epochs and batch size
3. The pipeline will:
   - Download the Twitter Airline Sentiment dataset from Kaggle → S3
   - Preprocess and split into train/val/test
   - Launch a `ml.g4dn.xlarge` GPU training job on SageMaker
   - Train a weighted DistilBERT classifier with balanced class weights
   - Evaluate on held-out test set (accuracy, F1, precision, recall)
   - Package with custom inference handler and deploy to your endpoint

### GitHub Secrets required for ML pipeline

| Secret | Value |
|---|---|
| `AWS_ROLE_ARN` | IAM role ARN for GitHub OIDC (no static key needed) |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN |
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_KEY` | Your Kaggle API key |

### GitHub Variables required

| Variable | Example |
|---|---|
| `AWS_REGION` | `ap-south-1` |
| `S3_BUCKET` | `your-sentient-bucket` |
| `PROJECT_NAME` | `customer-sentiment-analysis-mlops` |
| `SAGEMAKER_ENDPOINT_NAME` | `customer-sentiment-analysis-mlops-endpoint` |

---

## CI/CD

All workflows use **GitHub OIDC for AWS authentication** — no long-lived IAM user keys are stored in GitHub Secrets for infrastructure operations.

| Workflow | Trigger | What it does |
|---|---|---|
| `01-ci.yml` | Push / PR | Secret scan (Gitleaks) · Python lint/test · Frontend build · Docker build |
| `02-deploy.yml` | Push to `main` | Builds ECR images · Deploys to ECS via OIDC |
| `03-ml-train-deploy.yml` | Manual | Full SageMaker train → deploy pipeline |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single text sentiment analysis |
| `POST` | `/api/v1/predict/batch` | Batch analysis (up to 500 texts) |
| `POST` | `/api/v1/predict/csv` | CSV file upload analysis (up to 10k rows) |
| `GET` | `/api/v1/analytics/overview` | Executive KPI summary |
| `GET` | `/api/v1/analytics/timeseries` | Sentiment trend time-series |
| `GET` | `/api/v1/analytics/attributes` | Attribute heatmap data |
| `GET` | `/api/v1/analytics/urgency` | Urgency trend breakdown |
| `GET` | `/api/v1/analytics/drift` | Model drift PSI/KS history |
| `GET` | `/api/v1/analytics/clusters` | Root-cause cluster insights |
| `GET` | `/api/v1/analytics/business-impact` | Scenario simulation |
| `POST` | `/api/v1/auth/login` | JWT token pair |
| `GET` | `/api/v1/health` | Deep health check |
| `GET` | `/health` | Liveness probe |

Full interactive documentation: `http://localhost:8000/docs`

---

## Observability

Sentient ships with a complete observability stack:

- **Prometheus** — metrics endpoint at `/metrics`, scraped every 15 seconds
- **Structured logging** — JSON-formatted with automatic credential redaction
- **Sentry** — exception tracking with environment tagging (set `SENTRY_DSN`)
- **Health checks** — liveness at `/health`, deep check at `/api/v1/health` (includes SageMaker endpoint status)
- **Request timing** — `X-Process-Time-Ms` header on every response
- **Celery** — task monitoring via Flower (add `flower` service to `docker-compose.yml`)

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Training | PyTorch · HuggingFace Transformers · DistilBERT |
| ML Serving | AWS SageMaker (managed endpoints) |
| Backend | FastAPI · SQLAlchemy async · Alembic · Celery |
| Database | PostgreSQL 16 |
| Cache / Queue | Redis 7 |
| Frontend | Next.js 15 · React 19 · TypeScript · Tailwind CSS |
| Charts | Recharts |
| Auth | JWT (python-jose) · bcrypt |
| Cloud | AWS (SageMaker · S3 · ECR · ECS · Secrets Manager) |
| Observability | Prometheus · Sentry · Structured logging |
| CI/CD | GitHub Actions · Docker · AWS OIDC |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Ensure `docker compose up` starts cleanly with a valid `.env`
4. Run CI locally: `ruff check apps/api/ && cd apps/web && npm run build`
5. Confirm no secrets are present: `git diff --staged | grep -E "AKIA|password"`
6. Open a pull request against `develop`

---

## Security

Sentient was designed with security as a first-class architectural principle:

- **No secrets in source** — enforced by Gitleaks in every CI run
- **Runtime-only secret loading** — credentials read from Secrets Manager or `.env`, never baked into images
- **Non-root Docker containers** — API and web run as unprivileged users
- **HTTPS-only in production** — enforce via ALB/CloudFront TLS termination
- **RBAC** — every tenant operation validates the caller's role
- **Rate limiting** — per-IP at the API gateway layer
- **Redacting log filter** — strips credential patterns from all log output

To report a security vulnerability, email security@your-org.com.

---

## License

MIT © Sentient Intelligence Platform Contributors

---

<div align="center">
<sub>Built with precision. Deployed with confidence. Trusted at scale.</sub>
</div>
