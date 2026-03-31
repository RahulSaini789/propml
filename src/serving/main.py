"""
Layer 6 — FastAPI Model Serving (Cloud Edition)
PropML: Property Price Prediction API

Architecture:
    Client → Render (Nginx/Uvicorn) → FastAPI → Hugging Face Model → SHAP → Response
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import shap
import pickle
import time
import uuid
import json
import logging
import os

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ──
# Pulls the repo name from Render Environment Variable, defaults to your HF repo
HF_REPO_ID = os.getenv("HF_MODEL_REPO", "Dumdigi/propml-gurgaon")

# ══════════════════════════════════════════
#  APP STATE
# ══════════════════════════════════════════

class AppState:
    models: dict     = {}   # city → xgb model
    explainers: dict = {}   # city → shap TreeExplainer
    feature_cols: dict = {} # city → list of feature column names
    start_time: float = 0
    prediction_count: int = 0
    latency_history: list = []

app_state = AppState()

# ══════════════════════════════════════════
#  LIFESPAN — Startup (Cloud Load) + Shutdown
# ══════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    app_state.start_time = time.time()
    logger.info("PropML API starting up in Cloud Mode...")
    
    city = "gurgaon"
    
    try:
        logger.info(f"Downloading model brain from HuggingFace: {HF_REPO_ID}")
        
        # 1. Download & Load Model
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pkl")
        with open(model_path, "rb") as f:
            app_state.models[city] = pickle.load(f)
            
        # 2. Download & Load Metadata
        meta_path = hf_hub_download(repo_id=HF_REPO_ID, filename="feature_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            # Match the exact key from your training script ("feature_columns" or "features")
            app_state.feature_cols[city] = metadata.get("feature_columns", metadata.get("features", []))
            
        # 3. Setup SHAP Explainer
        app_state.explainers[city] = shap.TreeExplainer(app_state.models[city])
        
        logger.info(f"Startup complete. Models loaded safely for: {list(app_state.models.keys())}")
        
    except Exception as e:
        logger.error(f"Critical Error loading model from HF: {e}")
        logger.warning("API will start but predictions will fail. Check HF_MODEL_REPO and token.")

    yield

    # ── SHUTDOWN ──
    app_state.models.clear()
    app_state.feature_cols.clear()
    app_state.explainers.clear()
    logger.info("PropML API shut down. Cloud memory cleared.")

# ══════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════

app = FastAPI(
    title="PropML API (Cloud)",
    description="Property price prediction for Indian real estate — Gurgaon (v1)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════

class PredictRequest(BaseModel):
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
        supported = {"gurgaon"}
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
    impact:     float
    direction:  str
    shap_value: float

class ConfidenceInterval(BaseModel):
    low:  float
    high: float
    note: str = "±15% uncertainty band based on CV error"

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
    relative_floor = 0.5
    if req.floor_pos and req.total_floors and req.total_floors > 0:
        relative_floor = min(req.floor_pos / req.total_floors, 1.0)

    bath_per_bed = min(req.bathroom / req.bedRoom, 3.0)
    log_area     = np.log1p(req.area_sqft)
    is_house     = 1 if req.property_type == "house" else 0

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
        "sector_encoded":    req.sector_encoded or 2.0,
    }

    row = {col: feature_dict.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row])

# ══════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    city = req.city

    if city not in app_state.models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for city '{city}' is not loaded. Check /health."
        )

    model        = app_state.models[city]
    explainer    = app_state.explainers[city]
    feature_cols = app_state.feature_cols[city]

    X = _build_feature_vector(req, feature_cols)

    log_pred = float(model.predict(X)[0])
    pred_cr  = round(float(np.expm1(log_pred)), 3)

    ci_low  = round(pred_cr * 0.85, 3)
    ci_high = round(pred_cr * 1.15, 3)

    shap_vals = explainer.shap_values(X)[0]
    total_shap_magnitude = float(np.abs(shap_vals).sum())
    top3_idx = np.argsort(np.abs(shap_vals))[::-1][:3]

    shap_features = [
        SHAPFeature(
            feature    = feature_cols[i],
            impact     = round(float(abs(shap_vals[i])) / total_shap_magnitude, 3) if total_shap_magnitude > 0 else 0,
            direction  = "positive" if shap_vals[i] > 0 else "negative",
            shap_value = round(float(shap_vals[i]), 4),
        )
        for i in top3_idx
    ]

    price_per_sqft = int(pred_cr * 1e7 / req.area_sqft)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    request_id = f"req_{uuid.uuid4().hex[:8]}"
    app_state.prediction_count += 1
    app_state.latency_history.append(latency_ms)
    if len(app_state.latency_history) > 1000:
        app_state.latency_history = app_state.latency_history[-500:]

    logger.info(f"Prediction | id={request_id} | city={city} | pred={pred_cr}Cr | latency={latency_ms}ms")

    return PredictResponse(
        prediction_cr       = pred_cr,
        confidence_interval = ConfidenceInterval(low=ci_low, high=ci_high),
        price_per_sqft      = price_per_sqft,
        model_version       = f"propml-{city}/HuggingFace",
        shap_top_features   = shap_features,
        request_id          = request_id,
        latency_ms          = latency_ms,
        city                = city,
    )

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
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
    if city not in app_state.models:
        raise HTTPException(status_code=404, detail=f"Model for '{city}' not loaded.")
    return {
        "model_name":      f"propml-{city}",
        "source":          f"Hugging Face Hub ({HF_REPO_ID})",
        "stage":           "Production",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)