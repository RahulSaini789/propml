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
#  TEST GROUP 1 — DATA CLEANING
# ══════════════════════════════════════════

class TestCleaningPipeline:
    """Tests for src/cleaning/pipeline.py"""

    @pytest.fixture
    def sample_houses_df(self):
        """Minimal houses dataframe for testing."""
        return pd.DataFrame({
            "price":        ["5.25 Crore", "2.1 Crore", None, "0.0"],
            "area":         ["(242 sq.m.) Plot Area", "(50 sq.m.) Plot Area",
                             "(108 sq.m.) Plot Area", "(200 sq.m.) Plot Area"],
            "bedRoom":      ["5 Bedrooms", "21 Bedrooms", "3 Bedrooms", "2 Bedrooms"],
            "noOfFloor":    ["3 Floors", "2 Floors", "5 Floors", "4 Floors"],
            "property_type":["house"] * 4,
            "bathroom":     [3, 4, 2, 2],
            "balcony":      ["2 Balconies", "No Balcony", "1 Balcony", "3+ Balconies"],
        })

    @pytest.fixture
    def sample_flats_df(self):
        """Minimal flats dataframe for testing."""
        return pd.DataFrame({
            "price":     ["45 Lac", "1.47 Crore", "70 Lac", "2.0 Crore"],
            "area":      ["₹ 5,000/sq.ft.", "₹ 7,692/sq.ft.",
                          "₹ 6,722/sq.ft.", "₹ 12,250/sq.ft."],
            "floorNum":  ["4th\xa0\xa0 of 4 Floors", "12nd\xa0\xa0 of 14 Floors",
                          "2nd\xa0\xa0 of 4 Floors", "5th\xa0\xa0 of 25 Floors"],
            "bedRoom":   [2, 3, 2, 4],
            "bathroom":  [2, 3, 2, 4],
            "balcony":   ["1 Balcony", "2 Balconies", "0", "3+ Balconies"],
            "property_type": ["flat"] * 4,
        })

    def test_houses_price_extracted_as_float(self, sample_houses_df):
        """Houses price strings should become floats in Crore."""
        from src.cleaning.pipeline import clean_houses_price
        result = clean_houses_price(sample_houses_df.copy())
        # Non-null prices should be floats
        non_null = result["price"].dropna()
        assert non_null.dtype in [float, np.float64]
        assert 5.25 in non_null.values
        assert 2.1  in non_null.values

    def test_houses_area_converted_to_sqft(self, sample_houses_df):
        """Houses area should be in sqft (multiplied by 10.7639)."""
        from src.cleaning.pipeline import clean_houses_area
        result = clean_houses_area(sample_houses_df.copy())
        assert "area_sqft" in result.columns
        # 242 sq.m. × 10.7639 ≈ 2604.7 sq.ft.
        assert result["area_sqft"].iloc[0] == pytest.approx(242 * 10.7639, rel=0.01)

    def test_bedroom_outlier_removed(self, sample_houses_df):
        """21 bedrooms in a house should be filtered out."""
        from src.cleaning.pipeline import clean_bedrooms, remove_outliers
        df = clean_bedrooms(sample_houses_df.copy())
        # Convert price to float for outlier removal
        df["price"] = [5.25, 2.1, np.nan, 0.0]
        df["area_sqft"] = [2604, 538, 1162, 2153]
        result = remove_outliers(df)
        assert result["bedRoom"].max() <= 10

    def test_flats_price_lac_converted(self, sample_flats_df):
        """'45 Lac' should become 0.45 Crore."""
        from src.cleaning.pipeline import clean_flats_price
        result = clean_flats_price(sample_flats_df.copy())
        # 45 Lac = 0.45 Crore
        assert result["price"].iloc[0] == pytest.approx(0.45, rel=0.01)
        # 1.47 Crore stays 1.47
        assert result["price"].iloc[1] == pytest.approx(1.47, rel=0.01)

    def test_floor_extraction(self, sample_flats_df):
        """Floor numbers extracted correctly from messy strings."""
        from src.cleaning.pipeline import clean_floors
        result = clean_floors(sample_flats_df.copy())
        assert "floor_pos"    in result.columns
        assert "total_floors" in result.columns
        # '4th of 4 Floors' → floor_pos=4, total_floors=4
        assert result["floor_pos"].iloc[0]    == 4.0
        assert result["total_floors"].iloc[0] == 4.0
        # floorNum column should be dropped
        assert "floorNum" not in result.columns

    def test_null_price_dropped(self):
        """Rows with null price should be removed before IQR."""
        from src.cleaning.pipeline import remove_outliers
        df = pd.DataFrame({
            "price":        [1.0, 2.0, np.nan, 1.5],
            "area_sqft":    [1000, 1500, 1200, 1100],
            "bedRoom":      [2, 3, 2, 2],
            "property_type":["flat"] * 4,
        })
        result = remove_outliers(df)
        assert result["price"].isna().sum() == 0

    def test_area_outlier_removed(self):
        """Area outlier (8.7 million sqft) must be removed by IQR."""
        from src.cleaning.pipeline import remove_outliers
        df = pd.DataFrame({
            "price":        [1.0] * 10 + [1.5],
            "area_sqft":    [1000] * 10 + [8_711_989],   # extreme outlier
            "bedRoom":      [3] * 11,
            "property_type":["flat"] * 11,
        })
        result = remove_outliers(df)
        assert result["area_sqft"].max() < 1_000_000


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
        In CI: tests run AFTER training step.
        """
        try:
            from src.serving.main import app
            return TestClient(app)
        except Exception:
            pytest.skip("API not available (model not trained yet)")

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