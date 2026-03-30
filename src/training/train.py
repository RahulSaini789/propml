"""
Layer 5 — Model Training Pipeline (Version 2.0 - Stratified & Optimized)
PropML: Property Price Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import optuna
import shap
import json
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score # 🛠️ ADDED: StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # quiet optuna logs

# ── Paths ──
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models/current")
REPORTS_DIR  = Path("reports")

# ── MLflow config ──
MLFLOW_TRACKING_URI  = "http://localhost:5000"
EXPERIMENT_NAME      = "propml-gurgaon-price"
REGISTERED_MODEL     = "propml-gurgaon"

# ── Production thresholds ──
MAPE_THRESHOLD = 22.0   # MAPE > 15% → don't promote to production
R2_THRESHOLD   = 0.82   # R² < 0.82 → don't promote to production


# ══════════════════════════════════════════
#  BLOCK 1 — DATA LOADING
# ══════════════════════════════════════════

def load_features() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_parquet(FEATURES_DIR / "train.parquet")

    with open(FEATURES_DIR / "feature_metadata.json") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["log_price"]

    print(f"Loaded: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Target (log_price): mean={y.mean():.3f}, std={y.std():.3f}")

    return X, y, feature_cols


# ══════════════════════════════════════════
#  BLOCK 2 — METRICS
# ══════════════════════════════════════════

def compute_metrics(y_true_log: np.ndarray,
                    y_pred_log: np.ndarray) -> dict:
    y_true_cr = np.expm1(y_true_log)
    y_pred_cr = np.expm1(y_pred_log)

    mape  = mean_absolute_percentage_error(y_true_cr, y_pred_cr) * 100
    r2    = r2_score(y_true_log, y_pred_log)
    rmse  = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    med_err = float(np.median(np.abs(y_true_cr - y_pred_cr)))

    return {
        "mape":           round(mape, 4),
        "r2":             round(r2, 4),
        "rmse_log":       round(rmse, 4),
        "median_error_cr": round(med_err, 4),
    }


# ══════════════════════════════════════════
#  BLOCK 3 — BASELINE MODEL
# ══════════════════════════════════════════

def run_baseline(X: pd.DataFrame, y: pd.Series, stratify_col: pd.Series) -> dict: # 🛠️ CHANGED: Added stratify_col
    print("\n" + "─"*40)
    print("  Baseline: Ridge Regression (Stratified)")
    print("─"*40)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 🛠️ CHANGED: StratifiedKFold
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, stratify_col)): # 🛠️ CHANGED: stratify_col passed
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = np.asarray(pipeline.predict(X_val))
        metrics = compute_metrics(y_val.values, preds)
        fold_metrics.append(metrics)

    avg = {k: round(np.mean([m[k] for m in fold_metrics]), 4)
           for k in fold_metrics[0]}

    print(f"  CV MAPE : {avg['mape']:.2f}%")
    print(f"  CV R²   : {avg['r2']:.4f}")
    print(f"  → Baseline set. XGBoost must beat this.")

    return avg


# ══════════════════════════════════════════
#  BLOCK 4 — DEFAULT XGBOOST (PRE-TUNING)
# ══════════════════════════════════════════

def run_xgboost_default(X: pd.DataFrame, y: pd.Series, stratify_col: pd.Series) -> dict: # 🛠️ CHANGED: Added stratify_col
    print("\n" + "─"*40)
    print("  XGBoost (Default Parameters - Stratified)")
    print("─"*40)

    model = xgb.XGBRegressor(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,    
        reg_lambda        = 1.0,    
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = -1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 🛠️ CHANGED: StratifiedKFold
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, stratify_col)): # 🛠️ CHANGED: stratify_col passed
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        preds = np.asarray(model.predict(X_val))
        metrics = compute_metrics(y_val.values, preds)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.4f}")

    avg = {k: round(np.mean([m[k] for m in fold_metrics]), 4)
           for k in fold_metrics[0]}

    print(f"  → CV MAPE : {avg['mape']:.2f}%")
    print(f"  → CV R²   : {avg['r2']:.4f}")

    return avg


# ══════════════════════════════════════════
#  BLOCK 5 — OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════

def tune_with_optuna(X: pd.DataFrame,
                     y: pd.Series,
                     stratify_col: pd.Series, # 🛠️ CHANGED: Added stratify_col
                     n_trials: int = 100) -> dict:
    print("\n" + "─"*40)
    print(f"  Optuna Tuning ({n_trials} trials)")
    print("─"*40)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000),
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "tree_method"      : "hist",
            "random_state"     : 42,
            "n_jobs"           : -1,
        }

        model = xgb.XGBRegressor(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 🛠️ CHANGED: StratifiedKFold

        fold_mapes = []
        for tr_idx, val_idx in skf.split(X, stratify_col): # 🛠️ CHANGED: stratify_col passed
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds  = model.predict(X_val)
            metrics = compute_metrics(y_val.values, preds)
            fold_mapes.append(metrics["mape"])

        return float(np.mean(fold_mapes))

    study = optuna.create_study(
        direction  = "minimize",     
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=10)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_mape   = study.best_value

    print(f"  Best MAPE : {best_mape:.2f}%")
    print(f"  Best params: {best_params}")

    return best_params


# ══════════════════════════════════════════
#  BLOCK 6 — FINAL TRAINING + MLFLOW
# ══════════════════════════════════════════

def train_final_model(X: pd.DataFrame,
                      y: pd.Series,
                      stratify_col: pd.Series, # 🛠️ CHANGED: Added stratify_col
                      best_params: dict,
                      feature_cols: list[str],
                      baseline_metrics: dict) -> None:
    print("\n" + "─"*40)
    print("  Final Training + MLflow Logging")
    print("─"*40)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-optuna-tuned") as run:

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("baseline_mape", baseline_metrics["mape"])

        model = xgb.XGBRegressor(**best_params, tree_method="hist",
                                  random_state=42, n_jobs=-1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 🛠️ CHANGED: StratifiedKFold
        fold_metrics = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, stratify_col)): # 🛠️ CHANGED: stratify_col passed
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds   = model.predict(X_val)
            metrics = compute_metrics(y_val.values, preds)
            fold_metrics.append(metrics)

            mlflow.log_metrics({
                f"fold{fold+1}_mape": metrics["mape"],
                f"fold{fold+1}_r2":   metrics["r2"],
            })
            print(f"  Fold {fold+1}: MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.4f}")

        avg = {k: round(np.mean([m[k] for m in fold_metrics]), 4)
               for k in fold_metrics[0]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()}) # type: ignore

        print(f"\n  CV MAPE : {avg['mape']:.2f}%")
        print(f"  CV R²   : {avg['r2']:.4f}")
        print(f"  Baseline MAPE was: {baseline_metrics['mape']:.2f}%")
        improvement = baseline_metrics['mape'] - avg['mape']
        print(f"  Improvement over baseline: {improvement:.2f}%")
        mlflow.log_metric("improvement_over_baseline", round(improvement, 4))

        final_model = xgb.XGBRegressor(**best_params, tree_method="hist",
                                        random_state=42, n_jobs=-1)
        final_model.fit(X, y, verbose=False)

        print("\n  Computing SHAP values...")
        explainer   = shap.TreeExplainer(final_model)
        sample_X    = X.sample(min(300, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample_X)

        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        mlflow.log_dict(
            shap_importance.set_index("feature")["mean_abs_shap"].to_dict(), # type: ignore
            "shap_importance.json"
        )

        print("\n  Top 5 Features by SHAP:")
        for _, row in shap_importance.head(5).iterrows():
            print(f"    {row['feature']:30s} → {row['mean_abs_shap']:.4f}")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = REPORTS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                **avg,
                "run_id":       run.info.run_id,
                "n_features":   len(feature_cols),
                "n_train_rows": len(X),
                "improvement_over_baseline": round(improvement, 4),
            }, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        # 🛠️ FIXED: Define signature and input example
        from mlflow.models.signature import infer_signature
        input_example = X.head(1)
        signature = infer_signature(X, final_model.predict(X))

        # 🛠️ STEP 1: Log the model safely (without registering yet)
        mlflow.xgboost.log_model( # type: ignore
            xgb_model=final_model,
            artifact_path="model", 
            signature=signature,
            input_example=input_example
        )

        # 🛠️ STEP 2: Register the model explicitly using the Run ID
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODELS_DIR / "model.pkl")
        joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

        print(f"\n{'─'*40}")
        print("  Production Gate Check:")
        
        # 🛠️ CHANGED: Fixed the reporting logic so it correctly tells us WHICH metric failed
        mape_passed = avg["mape"] <= MAPE_THRESHOLD
        r2_passed = avg["r2"] >= R2_THRESHOLD
        
        if mape_passed:
            print(f"  PASSED ✅ MAPE {avg['mape']:.2f}% ≤ {MAPE_THRESHOLD}%")
        else:
            print(f"  FAILED ❌ MAPE {avg['mape']:.2f}% (threshold: {MAPE_THRESHOLD}%)")
            
        if r2_passed:
            print(f"  PASSED ✅ R²   {avg['r2']:.4f} ≥ {R2_THRESHOLD}")
        else:
            print(f"  FAILED ❌ R²   {avg['r2']:.4f} (threshold: {R2_THRESHOLD})")

        if mape_passed and r2_passed:
            print(f"  → Model ready for staging promotion")
        else:
            print(f"  → Model NOT promoted. Tune further.")

        print(f"\n  MLflow Run ID: {run.info.run_id}")
        print(f"  View: {MLFLOW_TRACKING_URI}/#/experiments/")

# ... (BLOCK 7 remains identical)

def explain_prediction(model_path: str,
                       feature_cols_path: str,
                       sample: dict) -> dict:
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)

    X_input = pd.DataFrame([sample])[feature_cols]

    log_pred   = model.predict(X_input)[0]
    pred_cr    = round(float(np.expm1(log_pred)), 3)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_input)[0]

    top3_idx   = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [
        {
            "feature":   feature_cols[i],
            "impact":    round(abs(shap_vals[i]) / sum(abs(shap_vals)), 2),
            "direction": "positive" if shap_vals[i] > 0 else "negative",
            "shap_value": round(float(shap_vals[i]), 4),
        }
        for i in top3_idx
    ]

    return {
        "prediction_cr":    pred_cr,
        "shap_top_features": top_features,
    }


# ══════════════════════════════════════════
#  MASTER PIPELINE
# ══════════════════════════════════════════

def run_training_pipeline(n_optuna_trials: int = 100): # 🛠️ CHANGED: Default trials updated to 100
    print("=" * 45)
    print("  Layer 5: Model Training Pipeline")
    print("=" * 45)

    X, y, feature_cols = load_features()
    
    # 🛠️ ADDED: Define stratify column based on 'is_house' feature
    stratify_col = X['is_house']

    baseline_metrics = run_baseline(X, y, stratify_col)
    default_metrics = run_xgboost_default(X, y, stratify_col)
    best_params = tune_with_optuna(X, y, stratify_col, n_trials=n_optuna_trials)
    train_final_model(X, y, stratify_col, best_params, feature_cols, baseline_metrics)

    print("\n" + "=" * 45)
    print("  Training Pipeline Complete!")
    print("=" * 45)


if __name__ == "__main__":
    # 🛠️ CHANGED: Running with 100 trials for a solid tuning phase
    run_training_pipeline(n_optuna_trials=30)
