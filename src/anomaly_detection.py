"""
anomaly_detection.py
--------------------
Trains and evaluates an Isolation Forest anomaly-detection model.

WHY Isolation Forest?
  The dataset has no fraud label, so supervised classification is not
  possible.  Isolation Forest is well-suited because:
    * It works on unlabelled data.
    * It handles mixed numeric features well.
    * Its anomaly score is continuous — perfect for a risk score.
    * It is computationally efficient on datasets of this size.

OUTPUT
  Each transaction gets:
    - anomaly_score  : raw Isolation Forest score (lower = more anomalous)
    - risk_score_pct : risk percentage 0–100 (higher = more suspicious)
    - risk_level     : "Low", "Medium", or "High"
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

logger = logging.getLogger(__name__)

# Risk thresholds — tuned to the actual Isolation Forest score distribution.
# The raw scores are min-max scaled to 0-100.  With contamination=0.08,
# the bulk of transactions cluster in the lower half of the scale.
# Thresholds are set at the 90th and 80th percentiles of the training scores
# so that ~10% → High Risk and ~10% → Medium Risk.
HIGH_RISK_THRESHOLD   = 50   # risk_score_pct >= this → High  (~top 10%)
MEDIUM_RISK_THRESHOLD = 37   # risk_score_pct >= this → Medium (~next 10%)


def build_model_pipeline(contamination: float = 0.08) -> Pipeline:
    """
    Build a sklearn Pipeline:
      1. StandardScaler  — centres and scales all features (required by IF)
      2. IsolationForest — the anomaly detection algorithm

    contamination: estimated fraction of anomalies in the data.
      0.08 means we expect ~8 % of transactions to be anomalous.
      Adjust this based on domain knowledge.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=200,       # more trees → more stable scores
            contamination=contamination,
            random_state=42,
            n_jobs=-1,              # use all CPU cores
        )),
    ])
    return pipeline


def train(df_features: pd.DataFrame, feature_cols: list) -> Pipeline:
    """
    Fit the Isolation Forest pipeline on the provided feature matrix.

    Parameters
    ----------
    df_features : DataFrame with engineered features
    feature_cols : list of column names to use as model input

    Returns
    -------
    fitted Pipeline
    """
    X = df_features[feature_cols].copy()

    # Safety: fill any remaining NaNs that slipped through preprocessing
    X = X.fillna(X.median())

    logger.info(f"Training Isolation Forest on {X.shape[0]:,} samples, "
                f"{X.shape[1]} features …")
    pipeline = build_model_pipeline()
    pipeline.fit(X)
    logger.info("Training complete.")
    return pipeline


def score_transactions(pipeline: Pipeline,
                       df_features: pd.DataFrame,
                       feature_cols: list) -> pd.DataFrame:
    """
    Apply the fitted pipeline to produce risk scores for every transaction.

    Isolation Forest's decision_function returns a score where:
      * more negative = more anomalous
      * close to 0 / positive = normal

    We rescale this to a 0-100 risk_score_pct where 100 = most anomalous.
    """
    df = df_features.copy()
    X  = df[feature_cols].fillna(df[feature_cols].median())

    # Raw scores: typically range from about -0.3 to +0.3
    raw_scores = pipeline.decision_function(X)

    # Invert: higher number = more anomalous
    inverted = -raw_scores

    # Min-max scale to [0, 100]
    min_s, max_s = inverted.min(), inverted.max()
    if max_s > min_s:
        risk_pct = ((inverted - min_s) / (max_s - min_s)) * 100
    else:
        risk_pct = np.zeros(len(inverted))

    df["anomaly_score_raw"] = raw_scores
    df["risk_score_pct"]    = np.round(risk_pct, 2)
    df["risk_level"]        = pd.cut(
        df["risk_score_pct"],
        bins=[-0.01, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 100.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    logger.info(
        f"Risk distribution:\n"
        f"  Low    : {(df['risk_level']=='Low').sum():,}\n"
        f"  Medium : {(df['risk_level']=='Medium').sum():,}\n"
        f"  High   : {(df['risk_level']=='High').sum():,}"
    )
    return df


def score_single(pipeline: Pipeline,
                 feature_vector: pd.DataFrame,
                 feature_cols: list,
                 all_scores: np.ndarray) -> dict:
    """
    Score a single transaction and return a dict with:
      risk_score_pct, risk_level, anomaly_score_raw

    all_scores: the raw inverted scores of ALL training transactions
      (needed to apply the same min-max scaling).
    """
    X = feature_vector[feature_cols].fillna(0)

    raw = pipeline.decision_function(X)[0]
    inverted = -raw

    # Apply the same min-max parameters from training
    min_s, max_s = all_scores.min(), all_scores.max()
    if max_s > min_s:
        risk_pct = ((inverted - min_s) / (max_s - min_s)) * 100
    else:
        risk_pct = 0.0

    risk_pct = float(np.clip(risk_pct, 0, 100))

    if risk_pct >= HIGH_RISK_THRESHOLD:
        level = "High"
    elif risk_pct >= MEDIUM_RISK_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    return {
        "risk_score_pct":    round(risk_pct, 2),
        "risk_level":        level,
        "anomaly_score_raw": round(float(raw), 6),
    }


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Pipeline saved → {path}")


def load_pipeline(path: str) -> Pipeline:
    pipeline = joblib.load(path)
    logger.info(f"Pipeline loaded ← {path}")
    return pipeline
