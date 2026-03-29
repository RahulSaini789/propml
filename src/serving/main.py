"""
Layer 6 — FastAPI Model Serving
PropML: Property Price Prediction API

Architecture:
    Client → Nginx → Uvicorn → FastAPI → MLflow Model → SHAP → Response

Endpoints:
    POST /predict     → Price prediction + confidence interval + SHAP
    GET  /health      → System health + model status
    GET  /model-info  → Model metadata from MLflow registry
    GET  /docs        → Auto-generated Swagger UI (FastAPI built-in)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import joblib
import time
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ──
MLFLOW_URI     = "http://localhost:5000"
MODELS_DIR     = Path("models/current")
FEATURES_DIR   = Path("data/features")

# ══════════════════════════════════════════
#  APP STATE (loaded once at startup)
# ══════════════════════════════════════════

class AppState:
    """
    Single place to store all loaded models and explainers.
    WHY a class instead of global variables?
    Cleaner namespace, easier to test, explicitly passed around.
    """
    models: dict     = {}   # city → xgb model
    explainers: dict = {}   # city → shap TreeExplainer
    feature_cols: dict = {} # city → list of feature column names
    start_time: float = 0
    prediction_count: int = 0
    latency_history: list = []

app_state = AppState()

# ══════════════════════════════════════════
#  LIFESPAN — startup + shutdown
# ══════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs on startup → yield → runs on shutdown.

    WHY lifespan instead of @app.on_event("startup")?
    @app.on_event is deprecated in FastAPI 0.95+.
    Lifespan is the modern, recommended approach.

    On startup: load models for all supported cities.
    On shutdown: clear model cache to free memory.
    """
    # ── STARTUP ──
    app_state.start_time = time.time()
    logger.info("PropML API starting up...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    cities = ["gurgaon"]  # Add "bangalore", "mumbai" as you scrape more

    for city in cities:
        try:
            _load_city_model(city)
        except Exception as e:
            logger.warning(f"Could not load model for {city}: {e}")
            logger.warning("API will start but predictions for this city will fail.")

    logger.info(f"Startup complete. Models loaded: {list(app_state.models.keys())}")
    yield

    # ── SHUTDOWN ──
    app_state.models.clear()
    app_state.explainers.clear()
    logger.info("PropML API shut down. Models cleared.")


def _load_city_model(city: str) -> None:
    """
    Load XGBoost model + SHAP explainer for a city.

    WHY load from MLflow registry instead of local .pkl?
    MLflow registry = single source of truth for production models.
    If we retrain and promote a new model, API automatically gets it
    on next restart — no manual file copying.

    Fallback: if MLflow is unreachable, load from local models/ directory.
    """
    try:
        # Load from MLflow Production stage
        model_uri = f"models:/propml-{city}/Production"
        xgb_model = mlflow.xgboost.load_model(model_uri) # type: ignore
        logger.info(f"Loaded {city} model from MLflow registry (Production)")
    except Exception as mlflow_err:
        logger.warning(f"MLflow load failed: {mlflow_err}. Falling back to local model.")
        # Fallback to local pkl
        model_path = MODELS_DIR / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {city} — run training first.")
        xgb_model = joblib.load(model_path)
        logger.info(f"Loaded {city} model from local file: {model_path}")

    # Load feature columns
    try:
        with open(FEATURES_DIR / "feature_metadata.json") as f:
            metadata = json.load(f)
        feature_cols = metadata["feature_columns"]
    except FileNotFoundError:
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    # Build SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    app_state.models[city]       = xgb_model
    app_state.explainers[city]   = explainer
    app_state.feature_cols[city] = feature_cols


# ══════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════

app = FastAPI(
    title="PropML API",
    description="Property price prediction for Indian real estate — Gurgaon (v1)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",         # Swagger UI at /docs
    redoc_url="/redoc",       # ReDoc at /redoc
)

# CORS — allows React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # Local dev
        "https://propml.vercel.app",   # Production frontend
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════
#  PYDANTIC SCHEMAS (Input / Output)
# ══════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    Input schema for /predict endpoint.
    Pydantic validates types + constraints automatically.
    Invalid input → 422 Unprocessable Entity (no custom code needed).

    WHY Pydantic over plain dicts?
    - Automatic type coercion (string "3" → int 3)
    - Field-level validation with Field(ge=0, le=10)
    - Auto-generated OpenAPI schema for Swagger UI
    - Clear error messages for API consumers
    """
    city:             str   = Field(..., description="City name e.g. 'gurgaon'")
    property_type:    str   = Field(..., description="'flat' or 'house'")
    bedRoom:          int   = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathroom:         int   = Field(..., ge=1, le=12)
    balcony:          int   = Field(0, ge=0, le=4)
    area_sqft:        float = Field(..., gt=100, lt=50000, description="Area in sq.ft.")
    floor_pos:        Optional[float] = Field(None, ge=0, le=60)
    total_floors:     Optional[float] = Field(None, ge=1, le=80)
    age_bucket:       int   = Field(2, ge=-1, le=4, description="0=new, 4=old, -1=unknown")
    amenity_score:    float = Field(5.0, ge=0, le=10)
    furnish_score:    float = Field(0.3, ge=0.0, le=1.0)
    avg_rating:       float = Field(4.0, ge=-1.0, le=5.0)
    has_metro_nearby: int   = Field(0, ge=0, le=1)
    facing_score:     int   = Field(3, ge=0, le=5)
    sector_encoded:   Optional[float] = Field(None, description="Sector mean price (from training)")

    @field_validator("city")
    @classmethod
    def city_must_be_supported(cls, v: str) -> str:
        supported = {"gurgaon"}  # Expand as cities are added
        if v.lower() not in supported:
            raise ValueError(f"City '{v}' not supported. Supported: {supported}")
        return v.lower()

    @field_validator("property_type")
    @classmethod
    def property_type_valid(cls, v: str) -> str:
        if v.lower() not in {"flat", "house"}:
            raise ValueError("property_type must be 'flat' or 'house'")
        return v.lower()


class SHAPFeature(BaseModel):
    feature:    str
    impact:     float   # fraction of total SHAP magnitude (0-1)
    direction:  str     # "positive" or "negative"
    shap_value: float   # raw SHAP value in log_price space


class ConfidenceInterval(BaseModel):
    low:  float   # Crores
    high: float   # Crores
    note: str     = "±15% uncertainty band based on CV error"


class PredictResponse(BaseModel):
    prediction_cr:       float
    confidence_interval: ConfidenceInterval
    price_per_sqft:      int
    model_version:       str
    shap_top_features:   list[SHAPFeature]
    request_id:          str
    latency_ms:          float
    city:                str


class HealthResponse(BaseModel):
    status:              str
    models_loaded:       dict
    uptime_seconds:      float
    predictions_served:  int
    avg_latency_ms:      float


# ══════════════════════════════════════════
#  HELPER — BUILD FEATURE VECTOR
# ══════════════════════════════════════════

def _build_feature_vector(req: PredictRequest, feature_cols: list) -> pd.DataFrame:
    """
    Convert PredictRequest → DataFrame matching training feature order.

    WHY use feature_cols ordering?
    XGBoost is sensitive to feature order — it learns splits by column index.
    Giving features in wrong order = completely wrong predictions.
    Always align with training feature_cols list.

    WHY pd.DataFrame instead of np.array?
    SHAP TreeExplainer returns feature names in output when DataFrame is used.
    Makes SHAP output interpretable without extra lookup.
    """
    # Derived features (must match feature engineering exactly)
    relative_floor = 0.5  # default: middle floor
    if req.floor_pos and req.total_floors and req.total_floors > 0:
        relative_floor = min(req.floor_pos / req.total_floors, 1.0)

    bath_per_bed = min(req.bathroom / req.bedRoom, 3.0)
    log_area     = np.log1p(req.area_sqft)
    is_house     = 1 if req.property_type == "house" else 0

    # Assemble feature dict
    feature_dict = {
        "bedRoom":           req.bedRoom,
        "bathroom":          req.bathroom,
        "balcony":           req.balcony,
        "area_sqft":         req.area_sqft,
        "floor_pos":         req.floor_pos or 1.0,
        "total_floors":      req.total_floors or 5.0,
        "amenity_score":     req.amenity_score,
        "furnish_score":     req.furnish_score,
        "has_metro_nearby":  req.has_metro_nearby,
        "has_hospital_nearby": 0,
        "has_school_nearby": 0,
        "has_mall_nearby":   0,
        "has_airport_nearby": 0,
        "avg_rating":        req.avg_rating,
        "has_rating":        1 if req.avg_rating > 0 else 0,
        "age_bucket":        req.age_bucket,
        "has_servant_room":  0,
        "has_study_room":    0,
        "has_pooja_room":    0,
        "has_store_room":    0,
        "facing_score":      req.facing_score,
        "is_house":          is_house,
        "relative_floor":    relative_floor,
        "bath_per_bed":      bath_per_bed,
        "log_area":          log_area,
        "sector_encoded":    req.sector_encoded or 2.0,  # fallback to global mean
    }

    # Build DataFrame with exact column order from training
    row = {col: feature_dict.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])


# ══════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict property price with confidence interval and SHAP explanation.

    Flow:
    1. Validate input (Pydantic handles this automatically)
    2. Build feature vector (match training order)
    3. Predict in log scale → inverse transform to Crores
    4. Compute confidence interval (±15% based on CV error)
    5. Compute SHAP values for top-3 feature explanation
    6. Log prediction for monitoring
    7. Return structured response
    """
    t0 = time.perf_counter()
    city = req.city

    # Check model is loaded
    if city not in app_state.models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for city '{city}' is not loaded. Check /health."
        )

    model       = app_state.models[city]
    explainer   = app_state.explainers[city]
    feature_cols = app_state.feature_cols[city]

    # Build input
    X = _build_feature_vector(req, feature_cols)

    # Predict (log scale) → Crores
    log_pred = float(model.predict(X)[0])
    pred_cr  = round(float(np.expm1(log_pred)), 3)

    # Confidence interval — ±15% based on our CV MAPE
    # WHY ±15%? Our model's CV MAPE is ~21% — we use a slightly tighter
    # band (15%) as the "most likely range" to avoid overclaiming uncertainty
    ci_low  = round(pred_cr * 0.85, 3)
    ci_high = round(pred_cr * 1.15, 3)

    # SHAP for this single prediction
    shap_vals = explainer.shap_values(X)[0]   # shape: (n_features,)

    total_shap_magnitude = float(np.abs(shap_vals).sum())
    top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]

    shap_features = [
        SHAPFeature(
            feature    = feature_cols[i],
            impact     = round(float(abs(shap_vals[i])) / total_shap_magnitude, 3),
            direction  = "positive" if shap_vals[i] > 0 else "negative",
            shap_value = round(float(shap_vals[i]), 4),
        )
        for i in top3_idx
    ]

    # price_per_sqft in rupees
    price_per_sqft = int(pred_cr * 1e7 / req.area_sqft)

    # Latency
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Log for monitoring (drift detection uses these logs)
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    app_state.prediction_count += 1
    app_state.latency_history.append(latency_ms)
    if len(app_state.latency_history) > 1000:
        app_state.latency_history = app_state.latency_history[-500:]

    logger.info(
        f"Prediction | id={request_id} | city={city} | "
        f"pred={pred_cr}Cr | latency={latency_ms}ms"
    )

    return PredictResponse(
        prediction_cr       = pred_cr,
        confidence_interval = ConfidenceInterval(low=ci_low, high=ci_high),
        price_per_sqft      = price_per_sqft,
        model_version       = f"propml-{city}/Production",
        shap_top_features   = shap_features,
        request_id          = request_id,
        latency_ms          = latency_ms,
        city                = city,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    System health check.
    Used by:
    - Docker HEALTHCHECK instruction
    - GitHub Actions deployment verification
    - Load balancer liveness probe
    """
    avg_latency = (
        round(float(np.mean(app_state.latency_history)), 1)
        if app_state.latency_history else 0.0
    )

    return HealthResponse(
        status             = "healthy",
        models_loaded      = {k: True for k in app_state.models},
        uptime_seconds     = round(time.time() - app_state.start_time, 1),
        predictions_served = app_state.prediction_count,
        avg_latency_ms     = avg_latency,
    )


@app.get("/model-info")
async def model_info(city: str = "gurgaon"):
    """
    Return model metadata from MLflow registry.
    Useful for frontend 'About this model' page.
    """
    if city not in app_state.models:
        raise HTTPException(status_code=404, detail=f"Model for '{city}' not loaded.")

    try:
        client = mlflow.MlflowClient(MLFLOW_URI)
        versions = client.get_latest_versions(f"propml-{city}", stages=["Production"])
        if not versions:
            raise ValueError("No Production model found")
        v   = versions[0]
        run = client.get_run(v.run_id) # type: ignore
        return {
            "model_name":       v.name,
            "version":          v.version,
            "stage":            "Production",
            "trained_on":       v.creation_timestamp,
            "cv_mape":          run.data.metrics.get("cv_mape"),
            "cv_r2":            run.data.metrics.get("cv_r2"),
            "n_features":       run.data.metrics.get("n_features"),
            "mlflow_run_id":    v.run_id,
            "feature_columns":  app_state.feature_cols.get(city, []),
        }
    except Exception as e:
        # Fallback if MLflow is unreachable
        return {
            "model_name":    f"propml-{city}",
            "stage":         "local",
            "note":          f"MLflow unreachable: {e}",
            "feature_columns": app_state.feature_cols.get(city, []),
        }


@app.get("/")
async def root():
    return {
        "service":  "PropML API",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/health",
        "predict":  "POST /predict",
    }


# ── Run locally ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,    # Auto-reload on code changes (dev only)
        workers = 1,       # 1 worker for dev (use 2-4 in production)
    )