"""
model_training.py
-----------------
Orchestrates the full training pipeline:
  1. Load raw data
  2. Clean
  3. Engineer features
  4. Train Isolation Forest
  5. Score all transactions
  6. Compute feature importances (permutation-based)
  7. Persist artefacts to disk

Run this file directly to (re)train the model:
    python src/model_training.py
"""

import os
import sys
import logging
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

# Make sure the src package is importable when run from any directory
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import load_raw_data, clean_data
from feature_engineering import build_features, MODEL_FEATURES
from anomaly_detection import train, score_transactions, save_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file → project root)
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_PATH   = os.path.join(BASE_DIR, "data", "raw",       "bank_transactions_data_2.csv")
PROC_DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "transactions_featured.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models",            "isolation_forest_pipeline.pkl")
SCORES_PATH     = os.path.join(BASE_DIR, "models",            "all_scores.npy")
FI_PATH         = os.path.join(BASE_DIR, "models",            "feature_importances.json")
META_PATH       = os.path.join(BASE_DIR, "models",            "model_meta.json")


def compute_feature_importance(pipeline, X: pd.DataFrame,
                                feature_names: list,
                                n_repeats: int = 10) -> dict:
    """
    Estimate feature importance via permutation importance.

    For anomaly detection we use the negative of the decision_function
    as the scoring function — higher means more anomalous.

    Returns a dict  {feature_name: mean_importance}  sorted descending.
    """
    logger.info("Computing permutation feature importances …")

    # We need a "scorer" compatible with permutation_importance.
    # Wrap the pipeline so it acts like an estimator with predict()
    class _Wrapper:
        def __init__(self, pipe):
            self.pipe = pipe
        def fit(self, X, y=None):
            return self
        def predict(self, X):
            return -self.pipe.decision_function(X)  # higher = more anomalous
        def score(self, X, y=None):
            return float(self.predict(X).mean())

    wrapper = _Wrapper(pipeline)

    # Use the anomaly score itself as the target proxy
    y_proxy = -pipeline.decision_function(X)

    result = permutation_importance(
        wrapper,
        X,
        y_proxy,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring=None,  # uses wrapper.score
    )

    fi = {
        name: float(imp)
        for name, imp in zip(feature_names, result.importances_mean)
    }
    # Sort descending
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    logger.info("Feature importances computed.")
    return fi


def run_training() -> None:
    """Full end-to-end training run."""

    # ----- 1. Load & clean -----
    raw_df   = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_data(raw_df)

    # ----- 2. Feature engineering -----
    featured_df = build_features(clean_df)

    # Persist the featured dataset for later SQL / EDA use
    os.makedirs(os.path.dirname(PROC_DATA_PATH), exist_ok=True)
    featured_df.to_csv(PROC_DATA_PATH, index=False)
    logger.info(f"Featured dataset saved → {PROC_DATA_PATH}")

    # ----- 3. Train -----
    pipeline = train(featured_df, MODEL_FEATURES)

    # ----- 4. Score all transactions -----
    scored_df = score_transactions(pipeline, featured_df, MODEL_FEATURES)

    # Save the raw inverted scores so the app can do correct min-max scaling
    # for any new single transaction
    raw_scores = -pipeline.decision_function(
        featured_df[MODEL_FEATURES].fillna(featured_df[MODEL_FEATURES].median())
    )
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    np.save(SCORES_PATH, raw_scores)
    logger.info(f"All raw scores saved → {SCORES_PATH}")

    # Persist the scored DataFrame back to processed folder
    scored_df.to_csv(PROC_DATA_PATH, index=False)

    # ----- 5. Save model -----
    save_pipeline(pipeline, MODEL_PATH)

    # ----- 6. Feature importances -----
    X_sample = featured_df[MODEL_FEATURES].fillna(0)
    # Use a sample for speed (permutation importance is slow on all 2 500 rows)
    sample_size = min(500, len(X_sample))
    X_samp = X_sample.sample(sample_size, random_state=42)

    fi = compute_feature_importance(pipeline, X_samp, MODEL_FEATURES, n_repeats=5)
    with open(FI_PATH, "w") as f:
        json.dump(fi, f, indent=2)
    logger.info(f"Feature importances saved → {FI_PATH}")

    # ----- 7. Model metadata -----
    meta = {
        "model_type":       "IsolationForest",
        "n_features":       len(MODEL_FEATURES),
        "feature_names":    MODEL_FEATURES,
        "n_train_samples":  int(len(featured_df)),
        "contamination":    0.08,
        "high_threshold":   50,
        "medium_threshold": 37,
        "risk_distribution": {
            "Low":    int((scored_df["risk_level"] == "Low").sum()),
            "Medium": int((scored_df["risk_level"] == "Medium").sum()),
            "High":   int((scored_df["risk_level"] == "High").sum()),
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model metadata saved → {META_PATH}")

    # Summary
    total = len(scored_df)
    high  = meta["risk_distribution"]["High"]
    med   = meta["risk_distribution"]["Medium"]
    low   = meta["risk_distribution"]["Low"]
    logger.info(
        f"\n{'='*50}\n"
        f"Training complete!\n"
        f"  Total transactions : {total:,}\n"
        f"  High Risk          : {high:,}  ({high/total*100:.1f}%)\n"
        f"  Medium Risk        : {med:,}  ({med/total*100:.1f}%)\n"
        f"  Low Risk           : {low:,}  ({low/total*100:.1f}%)\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    run_training()
