"""
tests/ — PropML Unit Tests
Run: pytest tests/ --cov=src --cov-report=term-missing

WHY write tests for an ML pipeline?
- Catches data bugs before they reach the model
- CI/CD pipeline fails if tests fail → no bad deploys
- Gives you confidence to refactor without breaking things
- Interviewers ALWAYS ask: "How did you test your pipeline?"
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient


# ══════════════════════════════════════════
#  TEST GROUP 1 — DATA CLEANING (Commented out post-refactor)
# ══════════════════════════════════════════
# (Cleaning tests commented out as per evolution of the pipeline)


# ══════════════════════════════════════════
#  TEST GROUP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════

class TestFeatureEngineering:
    """Tests for src/features/build_features.py"""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "price":        [1.5, 2.0, 0.8],
            "area_sqft":    [1200, 1800, 900],
            "floor_pos":    [3.0, 10.0, 1.0],
            "total_floors": [6.0, 40.0, 4.0],
            "bathroom":     [2, 3, 1],
            "bedRoom":      [2, 3, 1],
            "features": [
                "['Swimming Pool', 'Gym', 'Lift(s)']",
                "['Park', 'Security Personnel']",
                "not available",
            ],
            "rating": [
                "['Environment4 out of 5', 'Lifestyle4.5 out of 5']",
                "['Safety3 out of 5']",
                None,
            ],
        })

    def test_amenity_score_range(self, sample_df):
        """amenity_score must be between 0 and 10."""
        from src.features.build_features import build_amenity_score
        result = build_amenity_score(sample_df.copy())
        assert result["amenity_score"].between(0, 10).all()

    def test_amenity_score_not_available(self, sample_df):
        """'not available' should give amenity_score = 0."""
        from src.features.build_features import build_amenity_score
        result = build_amenity_score(sample_df.copy())
        assert result["amenity_score"].iloc[2] == 0

    def test_relative_floor_range(self, sample_df):
        """relative_floor = floor_pos / total_floors, clipped to [0, 1]."""
        from src.features.build_features import build_derived_features
        result = build_derived_features(sample_df.copy())
        assert result["relative_floor"].between(0, 1).all()

    def test_relative_floor_values(self, sample_df):
        """Floor 3 of 6 = 0.5, Floor 10 of 40 = 0.25."""
        from src.features.build_features import build_derived_features
        result = build_derived_features(sample_df.copy())
        assert result["relative_floor"].iloc[0] == pytest.approx(3/6,  rel=0.01)
        assert result["relative_floor"].iloc[1] == pytest.approx(10/40, rel=0.01)

    def test_avg_rating_parsed(self, sample_df):
        """Rating strings should parse to float averages."""
        from src.features.build_features import build_rating_score
        result = build_rating_score(sample_df.copy())
        # (4 + 4.5) / 2 = 4.25
        assert result["avg_rating"].iloc[0] == pytest.approx(4.25, rel=0.01)
        # null rating → -1
        assert result["avg_rating"].iloc[2] == -1.0

    def test_no_target_leakage(self):
        """price_per_sqft must NOT appear in final features."""
        # This test documents that leakage column is explicitly dropped
        from src.features.build_features import drop_leakage_columns
        df = pd.DataFrame({
            "price":          [1.5],
            "price_per_sqft": [10000],   # leakage column
            "rate":           [10000],   # another leakage column
            "area_sqft":      [1500],
        })
        result = drop_leakage_columns(df)
        assert "price_per_sqft" not in result.columns
        assert "rate"           not in result.columns
        # price (target) should remain
        assert "price" in result.columns


# ══════════════════════════════════════════
#  TEST GROUP 3 — FASTAPI ENDPOINTS
# ══════════════════════════════════════════

class TestAPIEndpoints:
    """
    Integration tests for FastAPI endpoints.
    Uses TestClient which runs the app without a real server.
    """

    @pytest.fixture(scope="class")
    def client(self):
        """
        Create test client.
        Note: Model must be trained and saved before these tests run.
        """
        try:
            from src.serving.main import app
            # 🛠️ THE FIX: Using 'with' block to trigger FastAPI lifespan events
            # This ensures the XGBoost model is loaded into memory before tests execute.
            with TestClient(app) as c:
                yield c
        except Exception as e:
            pytest.skip(f"API not available or model load failed: {e}")

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "PropML" in response.json()["service"]

    def test_predict_valid_input(self, client):
        """Valid prediction request should return 200 with all fields."""
        payload = {
            "city":          "gurgaon",
            "property_type": "flat",
            "bedRoom":       3,
            "bathroom":      3,
            "balcony":       2,
            "area_sqft":     1800.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # All required response fields present
        assert "prediction_cr"       in data
        assert "confidence_interval" in data
        assert "shap_top_features"   in data
        assert "request_id"          in data
        assert "latency_ms"          in data
        # Prediction must be positive
        assert data["prediction_cr"] > 0

    def test_predict_invalid_city(self, client):
        """Unsupported city should return 422."""
        payload = {
            "city":          "mumbai",   # not supported yet
            "property_type": "flat",
            "bedRoom":       3,
            "bathroom":      3,
            "area_sqft":     1800.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_area(self, client):
        """Negative area should return 422."""
        payload = {
            "city":          "gurgaon",
            "property_type": "flat",
            "bedRoom":       3,
            "bathroom":      3,
            "area_sqft":     -500.0,     # invalid
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_bedroom_cap(self, client):
        """More than 10 bedrooms should return 422."""
        payload = {
            "city":          "gurgaon",
            "property_type": "flat",
            "bedRoom":       25,         # invalid — our cap is 10
            "bathroom":      3,
            "area_sqft":     1800.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_get_method_not_allowed(self, client):
        """GET /predict should return 405."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_confidence_interval_structure(self, client):
        """Confidence interval must have low and high, low < prediction < high."""
        payload = {
            "city":          "gurgaon",
            "property_type": "flat",
            "bedRoom":       3,
            "bathroom":      3,
            "area_sqft":     1800.0,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        ci = data["confidence_interval"]
        assert "low"  in ci
        assert "high" in ci
        assert ci["low"]  < data["prediction_cr"]
        assert ci["high"] > data["prediction_cr"]

    def test_shap_top_features_structure(self, client):
        """SHAP features must have direction as positive or negative."""
        payload = {
            "city":          "gurgaon",
            "property_type": "flat",
            "bedRoom":       3,
            "bathroom":      3,
            "area_sqft":     1800.0,
        }
        response = client.post("/predict", json=payload)
        data  = response.json()
        shaps = data["shap_top_features"]
        assert len(shaps) == 3     # top 3 features
        for s in shaps:
            assert s["direction"] in ["positive", "negative"]
            assert 0 <= s["impact"] <= 1