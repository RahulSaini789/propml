# PropML — Property Price Prediction

<div align="center">

![PropML Banner](https://img.shields.io/badge/PropML-Gurgaon%20Real%20Estate%20Intelligence-c9a84c?style=for-the-badge&labelColor=0d1117)

[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-10b981?style=flat-square)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=flat-square)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=flat-square)](https://dvc.org)

**Production-grade MLOps pipeline for Gurgaon property price prediction.**
From raw scraping to live API serving — 7 layers, end-to-end.

[Live Demo](#-live-demo) · [API Docs](#-api-reference) · [Architecture](#-architecture) · [Quick Start](#-quick-start)

</div>

---

## 🎯 What Is This?

PropML is a **full production MLOps project** — not a Jupyter notebook. It predicts Gurgaon property prices using XGBoost trained on 3,417 real properties scraped from 99acres and Housing.com.

**What makes it production-grade:**
- City-aware data cleaning pipeline (not just `df.dropna()`)
- 13 engineered features including K-Fold target-encoded sector
- Bayesian hyperparameter tuning with Optuna (100 trials)
- SHAP explainability on every prediction — not just a black box number
- MLflow model registry with quality gate (MAPE < 15% before promotion)
- Multi-stage Docker build at 680MB
- GitHub Actions CI/CD with 5-stage pipeline

---

## 🚀 Live Demo

| Service | URL | Description |
|---------|-----|-------------|
| **Website** | [propml.vercel.app](https://propml.vercel.app) | Price predictor + market insights |
| **API** | [propml-api.onrender.com](https://propml-api.onrender.com) | REST API (may sleep on free tier) |
| **API Docs** | [/docs](https://propml-api.onrender.com/docs) | Swagger UI — try it live |
| **Health** | [/health](https://propml-api.onrender.com/health) | Model status + uptime |

> **Note:** Free tier on Render sleeps after 15 min inactivity. First request takes ~30s to wake up.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│  99acres ──┐                                                        │
│            ├──► Scrapy ──► data/raw/ ──► DVC (S3 storage)          │
│  Housing ──┘              (houses.csv + flats.csv)                  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                               │
│                                                                     │
│  Great Expectations  ──►  Schema validation per city                │
│                                                                     │
│  Cleaning Pipeline:                                                 │
│    houses.csv ──► clean_houses_price/area/floors ──┐               │
│    flats.csv  ──► clean_flats_price/area/floors  ──┼──► master.parquet│
│                   (grouped IQR outlier removal)    │               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                        │
│                                                                     │
│  Raw Columns (20)  ──►  Engineered Features (13 new)               │
│                                                                     │
│  amenity_score     = weighted sum (pool=3, gym=2, lift=1...)       │
│  furnish_score     = appliances present / total possible            │
│  relative_floor    = floor_pos / total_floors (clipped 0-1)        │
│  bath_per_bed      = bathroom / bedRoom (luxury proxy)             │
│  avg_rating        = regex parse "4.5 out of 5" → float           │
│  has_metro_nearby  = keyword search in nearbyLocations             │
│  age_bucket        = ordinal 0-4 (Under Construction → 10yr+)     │
│  sector_encoded    = K-Fold target encoding (5-fold, smoothed)     │
│  log_area          = log1p(area_sqft) for skew correction         │
│  log_price (target)= log1p(price) for skew correction             │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                                 │
│                                                                     │
│  Ridge Baseline ──► XGBoost (default) ──► Optuna (100 trials)     │
│                                                 │                   │
│              5-Fold CV ◄────────────────────────┘                  │
│                  │                                                  │
│  MLflow Tracking:                                                   │
│    params: n_estimators, max_depth, learning_rate...               │
│    metrics: cv_mape, cv_r2, fold1_mape...fold5_mape               │
│    artifacts: shap_importance.json, metrics.json                   │
│    model: registered to propml-gurgaon/Production                 │
│                                                                     │
│  Quality Gate: MAPE < 15% AND R² > 0.82 → promote to Production   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                                 │
│                                                                     │
│  Client ──► Nginx ──► Uvicorn (2 workers) ──► FastAPI              │
│                                                   │                │
│  POST /predict:                                   ▼                │
│    1. Pydantic validation (area>100, bedRoom≤10...)               │
│    2. Build feature vector (match training order)                  │
│    3. XGBoost predict (log scale) → expm1 → Crores               │
│    4. SHAP TreeExplainer → top-3 features + direction             │
│    5. Confidence interval (±15% based on CV error)               │
│    6. Log prediction for drift monitoring                         │
│    7. Return structured JSON response                              │
│                                                                    │
│  GET /health   → Docker healthcheck + uptime + latency stats      │
│  GET /model-info → MLflow registry metadata                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CI/CD LAYER                                    │
│                                                                     │
│  git push ──► GitHub Actions:                                      │
│                                                                     │
│  [lint] ──► [test] ──► [model-gate] ──► [docker] ──► [deploy]     │
│    30s        2min        5s             5min          2min        │
│                                                                     │
│  model-gate: reads reports/metrics.json                           │
│              MAPE > 15% → sys.exit(1) → pipeline stops           │
│              MAPE ≤ 15% → Docker build proceeds                   │
│                                                                     │
│  Smoke test after deploy: /health 200 + /predict sanity check     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Baseline (Ridge) | XGBoost (Default) | XGBoost (Tuned) |
|--------|-----------------|-------------------|-----------------|
| CV MAPE | 34.45% | 21.33% | **21.03%** |
| CV R² | 0.7345 | 0.8507 | **0.8493** |
| Improvement | — | +13.12% | **+13.42%** |

**Top SHAP Features (Global Importance):**

```
is_house         ████████████████████ 0.193
area_sqft        █████████████        0.130
avg_rating       ████                 0.041
total_floors     ████                 0.040
bathroom         ███                  0.036
```

> **Why is_house is #1?** Gurgaon has two completely different sub-markets.
> Independent houses include land value — commanding 2-3x premium over flats.
> The model correctly identifies this as the dominant price driver.

---

## ⚡ Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone repo
git clone https://github.com/rahulsaini/propml.git
cd propml

# Create environment file
echo "DB_PASSWORD=propml_secure_123" > .env

# Start full stack (Postgres + MLflow + API + Grafana)
docker-compose up --build

# Wait ~60 seconds for model to load, then test:
curl http://localhost:8000/health
```

### Option 2 — Local Development

```bash
# Clone + setup
git clone https://github.com/rahulsaini/propml.git
cd propml
conda create -n propml python=3.11 -y
conda activate propml
pip install -r requirements.txt

# Pull data from DVC remote
dvc pull

# Run full pipeline (clean → features → train)
dvc repro

# Start API
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔍 Model Validation — How to Verify It Works

### Step 1 — Health Check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "models_loaded": {"gurgaon": true},
  "uptime_seconds": 120.5,
  "predictions_served": 0,
  "avg_latency_ms": 0.0
}
```

### Step 2 — Sanity Predictions

Test with known property types and verify predictions make logical sense:

```bash
# Test 1: Small 1BHK flat (expect ~0.4-0.8 Cr)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"gurgaon","property_type":"flat","bedRoom":1,"bathroom":1,"balcony":0,"area_sqft":500}'

# Test 2: Standard 3BHK flat (expect ~1.2-2.0 Cr)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"gurgaon","property_type":"flat","bedRoom":3,"bathroom":3,"balcony":2,"area_sqft":1800}'

# Test 3: Large house (expect significantly more than flat of same area)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"gurgaon","property_type":"house","bedRoom":5,"bathroom":5,"balcony":3,"area_sqft":3000,"amenity_score":8}'
```

**Sanity checks to verify:**
- House price > flat price for same area ✓
- Larger area = higher price ✓
- Higher amenity_score = higher price ✓
- `confidence_interval.low` < `prediction_cr` < `confidence_interval.high` ✓

### Step 3 — SHAP Validation

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"gurgaon","property_type":"flat","bedRoom":3,"bathroom":3,"area_sqft":1800}' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Prediction: Rs {data[\"prediction_cr\"]} Cr')
print(f'Confidence: Rs {data[\"confidence_interval\"][\"low\"]} - Rs {data[\"confidence_interval\"][\"high\"]} Cr')
print(f'Latency: {data[\"latency_ms\"]}ms')
print('Top Factors:')
for f in data['shap_top_features']:
    arrow = '↑' if f['direction'] == 'positive' else '↓'
    print(f'  {arrow} {f[\"feature\"]:20s} {f[\"impact\"]*100:.1f}%')
"
```

### Step 4 — Run Unit Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Step 5 — Check MLflow Experiments

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open http://localhost:5000
# You should see:
# - Experiment: propml-gurgaon-price
# - Runs with cv_mape, cv_r2 logged
# - Model registered: propml-gurgaon
```

---

## 📁 Project Structure

```
propml/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                    # Immutable raw CSVs (DVC tracked)
│   ├── processed/              # Cleaned parquet files
│   └── features/               # Feature-engineered train.parquet
├── docker/
│   └── Dockerfile.api          # Multi-stage production build
├── models/
│   └── current/                # Latest trained model artifacts
│       ├── model.pkl
│       └── feature_cols.pkl
├── reports/
│   └── metrics.json            # CV metrics (read by CI gate)
├── src/
│   ├── cleaning/
│   │   └── pipeline.py         # City-aware data cleaning
│   ├── features/
│   │   └── build_features.py   # Feature engineering (13 features)
│   ├── training/
│   │   └── train.py            # XGBoost + Optuna + MLflow
│   └── serving/
│       └── main.py             # FastAPI application
├── tests/
│   └── test_pipeline.py        # pytest unit tests (15+ tests)
├── configs/
│   └── params.yaml             # Single source of truth for hyperparams
├── frontend/
│   └── index.html              # Production website
├── dvc.yaml                    # 5-stage DVC pipeline definition
├── docker-compose.yml          # Full stack: Postgres+MLflow+API+Grafana
└── requirements.txt            # Python dependencies
```

---

## 🔌 API Reference

### POST /predict

**Request:**
```json
{
  "city": "gurgaon",
  "property_type": "flat",
  "bedRoom": 3,
  "bathroom": 3,
  "balcony": 2,
  "area_sqft": 1800,
  "floor_pos": 10,
  "total_floors": 15,
  "age_bucket": 2,
  "amenity_score": 7.0,
  "furnish_score": 0.4,
  "has_metro_nearby": 1
}
```

**Response:**
```json
{
  "prediction_cr": 1.82,
  "confidence_interval": {
    "low": 1.55,
    "high": 2.09,
    "note": "±15% uncertainty band based on CV error"
  },
  "price_per_sqft": 10111,
  "model_version": "propml-gurgaon/Production",
  "shap_top_features": [
    {"feature": "area_sqft", "impact": 0.41, "direction": "positive", "shap_value": 0.2341},
    {"feature": "is_house",  "impact": 0.28, "direction": "negative", "shap_value": -0.1823},
    {"feature": "amenity_score", "impact": 0.14, "direction": "positive", "shap_value": 0.0921}
  ],
  "request_id": "req_a3f8b2c1",
  "latency_ms": 18.4,
  "city": "gurgaon"
}
```

### GET /health
```json
{
  "status": "healthy",
  "models_loaded": {"gurgaon": true},
  "uptime_seconds": 3600,
  "predictions_served": 142,
  "avg_latency_ms": 22.4
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data Versioning | DVC + S3 | Git tracks code, DVC tracks data |
| Validation | Great Expectations | City-specific data quality rules |
| Cleaning | Pandas + Regex | City-aware pipeline, grouped IQR |
| Features | scikit-learn + NLTK | K-Fold target encoding, NLP extraction |
| Model | XGBoost | Best for tabular mixed-type data |
| Tuning | Optuna (TPE) | Bayesian search, 10x faster than Grid |
| Tracking | MLflow | Experiment registry + model lifecycle |
| Explainability | SHAP | Local + global feature importance |
| Serving | FastAPI + Uvicorn | ASGI, async, auto-docs, Pydantic |
| Containerization | Docker (multi-stage) | 680MB image, non-root user |
| Orchestration | docker-compose | Postgres + MLflow + API + Grafana |
| CI/CD | GitHub Actions | Lint → Test → Gate → Docker → Deploy |
| Monitoring | Grafana + Prometheus | Latency, drift, throughput dashboards |

---

## 🔬 Key Engineering Decisions

**Why Grouped IQR instead of Global IQR?**
Global IQR on merged houses + flats data artificially pulled the upper fence down (flats dominated statistics), causing 54% of valid luxury houses to be deleted. Grouped IQR applies separate statistical bounds per property_type.

**Why K-Fold Target Encoding for sector?**
Naive target encoding creates data leakage — each row sees its own price when computing its sector's mean. K-Fold encoding uses out-of-fold means, ensuring no row sees its own target during encoding.

**Why log1p transform on price?**
Price distribution is right-skewed (range: 0.16 to 32 Cr). Training on raw prices causes MSE to overfit to expensive outliers. log1p creates symmetric distribution. At prediction: expm1(prediction) → Crores.

**Why MAPE over MSE as primary metric?**
MSE penalizes a 5Cr error on a luxury house 25x more than a 1Cr error on a budget flat. MAPE is scale-invariant — 10% error means the same thing regardless of price level. Business stakeholders understand "within 21% of actual price."

---

## 👨‍💻 Author

**Rahul Saini**
- B.Sc. Mathematics, University of Kota
- DSMP 2.0 Certification — CampusX
- Target: Data Scientist / MLOps Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/rahulsaini)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/rahulsaini)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**PropML** — Built with production engineering, not just model training.

*From raw scraped data to live API with CI/CD — 7 layers, fully documented.*

</div>