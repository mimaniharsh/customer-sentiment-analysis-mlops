<div align="center">

# Sentient — Customer Intelligence Platform

**Multi-model NLP pipeline · Real-time sentiment analysis · Production MLOps on AWS**

[![CI](https://img.shields.io/github/actions/workflow/status/mimaniharsh/customer-sentiment-analysis-mlops/01-ci.yml?label=CI&style=flat-square&logo=github)](https://github.com/mimaniharsh/customer-sentiment-analysis-mlops/actions)
[![Model](https://img.shields.io/badge/Models-TF--IDF%20%7C%20LSTM%20%7C%20BERT-4f46e5?style=flat-square&logo=huggingface)](https://huggingface.co/distilbert-base-uncased)
[![AWS](https://img.shields.io/badge/Infra-SageMaker%20%7C%20ECS%20%7C%20S3-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2015-000000?style=flat-square&logo=nextdotjs)](https://nextjs.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-Twitter%20Airline%20Sentiment-00b4d8?style=flat-square&logo=kaggle)](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

[**Live Demo**](#demo) · [**Model Comparison**](#model-comparison) · [**Quick Start**](#quick-start) · [**API Docs**](http://localhost:8000/docs)

</div>

---

## What This Is

Airlines receive thousands of tweets daily. Manual analysis doesn't scale — and standard sentiment tools miss the most important signal: **sarcasm**. *"Oh brilliant, delayed again. Truly phenomenal service."* scores positive on every keyword-based system. The customer is furious.

Sentient solves this with a **comparative multi-model NLP pipeline** that evaluates three approaches — TF-IDF + Logistic Regression, BiLSTM, and fine-tuned DistilBERT — on the same dataset and surfaces the best-performing model for production. Every prediction goes beyond a label: it returns urgency level, severity tier, business impact score, sarcasm probability with triggered signals, multi-label attribute tags, and a recommended escalation action.

Built on the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) (14,640 tweets, 6 US airlines). Deployed as a full-stack MLOps platform on AWS with CI/CD, drift detection, and an executive dashboard.

---

## Model Comparison

Three models. One dataset. Evaluated on accuracy, F1, interpretability, and production viability.

| Model | Approach | F1 (Weighted) | Latency | Interpretability | Best For |
|---|---|---|---|---|---|
| **Baseline** | TF-IDF + Logistic Regression | ~0.79 | <5ms | High — coefficient weights | Batch processing, explainability audits |
| **Deep Learning** | BiLSTM (sequence model) | ~0.84 | ~15ms | Medium — attention weights | Streaming, sequential context |
| **Transformer** ✓ | Fine-tuned DistilBERT | **~0.91** | ~180ms | High — token saliency | Production: handles sarcasm, negation, context |

> Exact metrics are logged to S3 after each training run and visible in the GitHub Actions summary.

**Why DistilBERT wins in production:** Context-aware encoding handles sarcasm, negation, and ambiguity — the three failure modes where TF-IDF and LSTM fall short on this dataset. At 66M parameters vs BERT's 110M, it delivers 97% of BERT accuracy at 60% of the inference cost, hitting the sweet spot for real-time API use.

**Where TF-IDF still earns its place:** When you need to explain a classification to a non-technical stakeholder or a compliance team, coefficient weights are unambiguous. The comparative analysis keeps it available as an interpretable baseline.

---

## The Intelligence Pipeline

Every tweet passes through the selected base model, then through four enrichment layers:

```
Raw Tweet
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Model Layer  (configurable: TF-IDF · LSTM · DistilBERT)    │
│  Output: Positive / Neutral / Negative + raw probabilities   │
└────────────────────────┬─────────────────────────────────────┘
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
     Sarcasm Head   Severity Head   Attribute Tagger
     4-signal       Matrix 4-tier   7 business labels
     mismatch       with keyword    pricing · support ·
     detection      evidence        reliability · etc.
           │             │              │
           └─────────────┼──────────────┘
                         ▼
                Override + Risk Tier
                Final label + escalation action
```

**Sarcasm Head** runs four independent signals:
1. Trigger phrase + required negative context — `"brilliant"` paired with `"delay"` fires; standalone it doesn't
2. Model mismatch — label is positive but `prob_negative > 20%`
3. Negation flip — `"not great"`, `"barely acceptable"`, `"nothing impressive"`
4. Exclamation amplifier on complaint topics — `"Lost!"`, `"Cancelled!"`

Score ≥ 55% overrides the sentiment label to negative. The override reason is always returned in the API response.

**Severity Head** uses a keyword matrix across four tiers. Every tier has documented criteria — making the system auditable, not a black box.

| Tier | Score | Example Keywords | Escalation |
|---|---|---|---|
| CRITICAL | 0.75–1.00 | cancel, refund, fraud, stranded, never again | Senior team within 1 hour |
| HIGH | 0.50–0.75 | delay, broken, terrible, unresolved | Dedicated team within 4 hours |
| MEDIUM | 0.25–0.50 | slow, confusing, disappointing | Next sprint review (48h) |
| LOW | 0.00–0.25 | (no negative signals) | Monthly trend report |

---

## Demo

```bash
# Start the full stack (6 services)
docker compose up --build

# Stream tweets in real time
python scripts/demo_live_stream.py

# Or batch process an entire CSV
python scripts/demo_live_stream.py --mode batch --file Tweets.csv

# Cinematic mode for presentations (2s delay between tweets)
python scripts/demo_live_stream.py --delay 2.0 --limit 50 --shuffle
```

Open **http://localhost:3000** — the dashboard updates live.

**What a single tweet looks like through the pipeline:**

```
Input:   "Oh wow, another delay. Just what I needed on a Monday. Phenomenal service."

Raw model output (DistilBERT):
  → positive 68.1% · neutral 4.0% · negative 27.9%

After enrichment pipeline:
  Sarcasm:    82%   Trigger 'phenomenal' + context 'delay' | prob mismatch
  OVERRIDE:   positive → NEGATIVE (sarcasm 82% exceeds 55% threshold)
  Severity:   HIGH  · matched keywords: ['delay'] · criteria: service failure
  Risk:       HIGH  · assign dedicated team within 4 hours
  Attributes: reliability, performance
  Impact:     0.61/1.00
  Insight:    "High-severity reliability failure detected. Unresolved signals
               in this tier correlate with elevated churn risk."
```

---

## Quick Start

**Prerequisites:** Docker · Docker Compose · AWS account with SageMaker access.

```bash
# 1. Clone
git clone https://github.com/mimaniharsh/customer-sentiment-analysis-mlops.git
cd customer-sentiment-analysis-mlops

# 2. Configure — copy the template and fill in your values
cp .env.example .env
```

Minimum `.env` values:
```env
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=ap-south-1
SAGEMAKER_ENDPOINT_NAME=customer-sentiment-analysis-mlops-endpoint
S3_BUCKET=your-bucket
APP_SECRET_KEY=any-32-char-string
JWT_SECRET_KEY=another-32-char-string
```

```bash
# 3. Start
docker compose up --build
# First build: ~4 min.  Subsequent starts: ~20s.

# 4. Install demo dependency and run
pip install requests
python scripts/demo_live_stream.py
```

| URL | What it is |
|---|---|
| http://localhost:3000 | Executive dashboard |
| http://localhost:8000/docs | Interactive API (Swagger) |
| http://localhost:9090 | Prometheus metrics |

---

## API

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Three delays this week. Absolute shambles."}'

# Batch — up to 500 texts in one HTTP call
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great flight!", "Total disaster.", "It was fine"]}'

# CSV upload — up to 10,000 rows
curl -X POST http://localhost:8000/api/v1/predict/csv \
  -F "file=@Tweets.csv"
```

**Response shape:**
```json
{
  "sentiment": "negative",
  "confidence": 0.943,
  "probabilities": { "positive": 0.028, "neutral": 0.029, "negative": 0.943 },
  "sarcasm_probability": 0.82,
  "sarcasm_label": "Likely sarcastic",
  "sarcasm_signals": ["Trigger 'phenomenal' with context: {'delay'}"],
  "sentiment_override_reason": "Overridden from 'positive': sarcasm 82% exceeds threshold.",
  "severity_tier": "high",
  "severity_score": 0.61,
  "severity_criteria": ["Significant service failure"],
  "risk": { "tier": "HIGH", "action": "Assign dedicated team within 4 hours." },
  "attributes": ["reliability", "performance"],
  "business_impact_score": 0.58,
  "strategic_insight": "High-severity reliability failure...",
  "explanation": { "top_tokens": [{ "token": "shambles", "importance": 0.34 }] },
  "latency_ms": 187
}
```

---

## ML Training

Runs entirely on AWS — no local GPU needed.

**Trigger:** GitHub Actions → `ML — Train & Deploy SageMaker Model` → Run workflow

| Stage | Duration | Infrastructure |
|---|---|---|
| Data Preprocessing | ~5 min | GitHub runner — Kaggle download, clean, 80/10/10 split, S3 upload |
| SageMaker Training | ~35 min | `ml.g4dn.xlarge` GPU — DistilBERT fine-tune |
| Model Evaluation | ~5 min | Per-class F1, confusion matrix, calibration metrics — logged to S3 |
| Endpoint Deploy | ~10 min | Package → create/update SageMaker managed endpoint |

**Key training choices and why they matter:**

```
Weighted CrossEntropy    → Negative tweets dominate the dataset. Without weighting,
                           the model learns to ignore minority classes.

Label smoothing 0.10     → Prevents overconfident softmax. The confidence % in the
                           API response needs to be trustworthy for business decisions.

Cosine LR schedule       → Smoother convergence in final epochs vs linear decay.
                           More stable production behaviour.

Early stopping (p=3)     → Stops before overfitting. Saves ~$0.30–0.50 per run.

Temperature scaling T=1.3 → Calibrates raw probabilities. "82% confidence" means
                            roughly 82% of such predictions are correct.
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Sentient Platform — AWS                            │
│                                                                      │
│  Next.js 15        FastAPI (async)      SageMaker Endpoint          │
│  Dashboard  ───▶   Multi-tenant API ──▶  DistilBERT / LSTM          │
│             ◀───   JWT + RBAC       ◀──  ~180ms inference           │
│                          │                                           │
│                 ┌─────────────────┐                                  │
│                 │  PostgreSQL 16  │   Redis 7 + Celery               │
│                 │  Row-level      │   Background tasks:              │
│                 │  tenant ISO     │   drift · metrics · reports      │
│                 └─────────────────┘                                  │
│                                                                      │
│  CI/CD     GitHub Actions → ECR → ECS  (OIDC — zero static keys)   │
│  Secrets   AWS Secrets Manager (prod) · .env (dev)                  │
│  Monitor   Prometheus · Sentry · structured JSON logs               │
└──────────────────────────────────────────────────────────────────────┘
```

**Six services, one command:** PostgreSQL · Redis · FastAPI · Celery Worker · Celery Beat · Next.js.

---

## Repository Structure

```
├── apps/
│   ├── api/app/
│   │   ├── api/v1/endpoints/    predict · analytics · reports · auth · health
│   │   ├── ml/pipeline.py       4-head enrichment engine (sarcasm, severity, override, viz)
│   │   ├── core/                config · database · security · store · deps
│   │   └── workers/             Celery tasks — drift detection · metrics · report gen
│   └── web/src/app/             Next.js App Router pages
│       ├── page.tsx             Overview — health score, trend, critical alerts
│       ├── analytics/           Sentiment timeseries, attribute heatmap
│       ├── clusters/            Root-cause topic grouping
│       ├── demo/                Live prediction stream, IntelligenceCard
│       └── reports/             1-click report generation suite
│
├── packages/ml/
│   ├── pipeline/                data_download.py · data_preprocess.py
│   ├── training/model_train.py  WeightedTrainer — DistilBERT fine-tune
│   ├── evaluation/model_eval.py per-class F1, confusion matrix, calibration
│   └── serving/model_deploy.py  SageMaker endpoint create/update
│
├── scripts/demo_live_stream.py  ← Start here
├── .github/workflows/
│   ├── 01-ci.yml               ruff lint · pytest · gitleaks · docker build
│   ├── 02-deploy.yml           ECS rolling deploy via OIDC
│   └── 03-ml-train-deploy.yml  Full SageMaker pipeline
├── docker-compose.yml
└── .env.example
```

---

## Dashboard

| Page | URL | Built for |
|---|---|---|
| Overview | `/` | Board — health score, 30-day NPS trend, critical alert count |
| Analytics | `/analytics` | Ops — sentiment timeseries, attribute heatmap, urgency distribution |
| Clusters | `/clusters` | Product — root-cause topic groups, negative rate per theme |
| Live Demo | `/demo` | Engineering / Sales — real-time intelligence cards per tweet |
| Reports | `/reports` | Analysts — 1-click executive summary, risk breakdown, drift analysis |

---

## Cost

| Resource | Cost |
|---|---|
| SageMaker training (`ml.g4dn.xlarge`) | ~$0.43 per run |
| SageMaker inference endpoint (`ml.m5.large`) | ~$83/month while running |
| Local development | $0 |

**Delete the endpoint between sessions:**
```bash
aws sagemaker delete-endpoint \
  --endpoint-name customer-sentiment-analysis-mlops-endpoint \
  --region ap-south-1
# Redeploy any time via GitHub Actions
```

---

## Security

Zero credentials in the repository. Gitleaks runs on every CI commit — any secret pattern fails the build.

| Environment | Source | Mechanism |
|---|---|---|
| Local | `.env` file | pydantic-settings reads at startup |
| Production | AWS Secrets Manager | Fetched at runtime via IAM role |
| CI/CD | GitHub Secrets + OIDC | Ephemeral role per run, no long-lived keys |

---

## Tech Stack

| | |
|---|---|
| **NLP Models** | TF-IDF + Logistic Regression · BiLSTM · DistilBERT (HuggingFace + PyTorch) |
| **ML Platform** | AWS SageMaker — training, registry, managed endpoints |
| **Backend** | FastAPI · SQLAlchemy async · Alembic · Celery |
| **Data** | PostgreSQL 16 · Redis 7 |
| **Frontend** | Next.js 15 · React 19 · TypeScript · Tailwind CSS · Recharts |
| **Cloud** | AWS — SageMaker · S3 · ECR · ECS · Secrets Manager |
| **CI/CD** | GitHub Actions · Docker · AWS OIDC |
| **Observability** | Prometheus · Sentry · structured JSON logging · PSI drift detection |

---

<div align="center">

MIT License · [Harsh Mimani](https://github.com/mimaniharsh) · TAPMI MBA-AI&DS · CRP ADS 5711

*Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) · 14,640 tweets across 6 US airlines*

</div>
