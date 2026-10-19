"""
prediction.py
-------------
Provides the functions called by the Streamlit app to:
  1. Load the trained model artefacts from disk.
  2. Build a feature vector from raw user input.
  3. Return a risk score + risk level + explainability factors.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import MODEL_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artefact paths — resolved relative to this file's location
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest_pipeline.pkl")
SCORES_PATH= os.path.join(BASE_DIR, "models", "all_scores.npy")
FI_PATH    = os.path.join(BASE_DIR, "models", "feature_importances.json")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")


# ---------------------------------------------------------------------------
# Artefact loading (cached between Streamlit reruns via st.cache_resource)
# ---------------------------------------------------------------------------

def load_artefacts() -> dict:
    """
    Load all model artefacts from disk.
    Returns a dict with keys: pipeline, all_scores, feature_importances, meta
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. "
            "Please run:  python src/model_training.py"
        )

    pipeline = joblib.load(MODEL_PATH)
    all_scores = np.load(SCORES_PATH)

    with open(FI_PATH) as f:
        feature_importances = json.load(f)

    with open(META_PATH) as f:
        meta = json.load(f)

    return {
        "pipeline":            pipeline,
        "all_scores":          all_scores,     # shape (n_train,) — inverted raw scores
        "feature_importances": feature_importances,
        "meta":                meta,
    }


# ---------------------------------------------------------------------------
# Build feature vector from a dict of raw transaction values
# ---------------------------------------------------------------------------

def build_feature_vector(tx_dict: dict,
                          df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dict of raw transaction field values (mirrors what a user would
    enter in the Streamlit form), return a single-row DataFrame with all
    MODEL_FEATURES filled in.

    For account-level aggregates we look up the existing history from
    df_full (the full featured dataset).  This mirrors what the training
    pipeline does and keeps the feature distributions consistent.

    Parameters
    ----------
    tx_dict : dict
        Must contain at minimum:
          TransactionAmount, TransactionDate (datetime or str),
          PreviousTransactionDate (datetime or str),
          TransactionType, Channel, CustomerOccupation,
          AccountID, DeviceID, MerchantID, Location,
          CustomerAge, TransactionDuration, LoginAttempts, AccountBalance

    df_full : DataFrame
        The full scored / featured DataFrame loaded by the Streamlit app.

    Returns
    -------
    single-row DataFrame with exactly the MODEL_FEATURES columns
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_engineering import (
        add_temporal_features,
        add_account_features,
        add_risk_indicators,
        encode_categoricals,
    )

    # Build a minimal one-row DataFrame
    row = {
        "TransactionID":          tx_dict.get("TransactionID", "TX_NEW"),
        "AccountID":              tx_dict["AccountID"],
        "TransactionAmount":      float(tx_dict["TransactionAmount"]),
        "TransactionDate":        pd.to_datetime(tx_dict["TransactionDate"]),
        "PreviousTransactionDate":pd.to_datetime(tx_dict["PreviousTransactionDate"]),
        "TransactionType":        tx_dict["TransactionType"],
        "Location":               tx_dict["Location"],
        "DeviceID":               tx_dict["DeviceID"],
        "IPAddress":              tx_dict.get("IPAddress", "0.0.0.0"),
        "MerchantID":             tx_dict["MerchantID"],
        "Channel":                tx_dict["Channel"],
        "CustomerAge":            float(tx_dict["CustomerAge"]),
        "CustomerOccupation":     tx_dict["CustomerOccupation"],
        "TransactionDuration":    float(tx_dict["TransactionDuration"]),
        "LoginAttempts":          int(tx_dict["LoginAttempts"]),
        "AccountBalance":         float(tx_dict["AccountBalance"]),
    }
    single_df = pd.DataFrame([row])

    # -- Temporal features --
    single_df = add_temporal_features(single_df)

    # -- Account aggregates --
    # Use existing history for this AccountID if available
    acct = tx_dict["AccountID"]
    acct_hist = df_full[df_full["AccountID"] == acct]

    if len(acct_hist) > 0:
        single_df["account_tx_count"]           = len(acct_hist)
        single_df["account_total_amount"]        = acct_hist["TransactionAmount"].sum()
        single_df["account_avg_amount"]          = acct_hist["TransactionAmount"].mean()
        single_df["account_std_amount"]          = acct_hist["TransactionAmount"].std() or 0
        single_df["account_n_devices"]           = acct_hist["DeviceID"].nunique()
        single_df["account_n_merchants"]         = acct_hist["MerchantID"].nunique()
        single_df["account_n_locations"]         = acct_hist["Location"].nunique()
        single_df["account_avg_login_attempts"]  = acct_hist["LoginAttempts"].mean()
    else:
        # Brand-new account — use dataset-wide medians as a safe fallback
        single_df["account_tx_count"]           = 1
        single_df["account_total_amount"]        = float(tx_dict["TransactionAmount"])
        single_df["account_avg_amount"]          = float(tx_dict["TransactionAmount"])
        single_df["account_std_amount"]          = 0.0
        single_df["account_n_devices"]           = 1
        single_df["account_n_merchants"]         = 1
        single_df["account_n_locations"]         = 1
        single_df["account_avg_login_attempts"]  = float(tx_dict["LoginAttempts"])

    # -- Risk indicators --
    # Compute global percentile thresholds from the full dataset
    p95_amount   = df_full["TransactionAmount"].quantile(0.95)
    p95_duration = df_full["TransactionDuration"].quantile(0.95)
    p05_balance  = df_full["AccountBalance"].quantile(0.05)
    p90_devices  = df_full["account_n_devices"].quantile(0.90)

    amt = float(tx_dict["TransactionAmount"])
    dur = float(tx_dict["TransactionDuration"])
    bal = float(tx_dict["AccountBalance"])
    log = int(tx_dict["LoginAttempts"])

    single_df["amount_to_balance_ratio"] = amt / (bal + 1)
    safe_std = single_df["account_std_amount"].iloc[0] or np.nan
    single_df["amount_z_score"] = (
        (amt - single_df["account_avg_amount"].iloc[0]) / safe_std
        if not np.isnan(safe_std) else 0.0
    )
    single_df["is_large_amount"]        = int(amt > p95_amount)
    single_df["is_long_duration"]       = int(dur > p95_duration)
    single_df["is_low_balance"]         = int(bal < p05_balance)
    single_df["is_high_login_attempts"] = int(log > 3)
    single_df["is_multi_device"]        = int(
        single_df["account_n_devices"].iloc[0] > p90_devices
    )

    # -- Encode categoricals --
    single_df = encode_categoricals(single_df)

    return single_df


# ---------------------------------------------------------------------------
# Explain which features contributed most for a single prediction
# ---------------------------------------------------------------------------

def explain_prediction(feature_vector: pd.DataFrame,
                        feature_importances: dict,
                        top_n: int = 5) -> list:
    """
    Return the top-N risk-contributing factors for a single transaction.

    Strategy:
      For each feature, multiply its global permutation importance by
      a "deviation" signal (how far the value is from the training median
      in the risk-increasing direction).  This gives a per-transaction
      per-feature risk weight.

    Returns a list of dicts:
      [{"feature": ..., "value": ..., "contribution": ...}, ...]
    """
    # Deviation weights (heuristic: abs z-score of the feature value)
    # We pre-normalised during training, so we just use the raw values here
    # weighted by global importance.
    contributions = []
    for feat, importance in feature_importances.items():
        if feat not in feature_vector.columns:
            continue
        val = feature_vector[feat].iloc[0]
        # Contribution = importance × |value|  (risk signals are positive)
        contribution = abs(float(importance)) * abs(float(val)) if pd.notna(val) else 0
        contributions.append({
            "feature":      feat,
            "value":        round(float(val), 4) if pd.notna(val) else None,
            "contribution": round(contribution, 6),
        })

    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions[:top_n]


FEATURE_LABELS = {
    "TransactionAmount":          "Transaction Amount",
    "TransactionDuration":        "Transaction Duration (s)",
    "LoginAttempts":              "Login Attempts",
    "AccountBalance":             "Account Balance",
    "CustomerAge":                "Customer Age",
    "tx_hour":                    "Transaction Hour",
    "tx_month":                   "Transaction Month",
    "tx_weekday":                 "Day of Week",
    "tx_is_weekend":              "Weekend Transaction",
    "hours_since_prev_tx":        "Hours Since Previous Tx",
    "account_tx_count":           "Account Transaction Count",
    "account_avg_amount":         "Account Avg. Amount",
    "account_std_amount":         "Account Std. Amount",
    "account_n_devices":          "Unique Devices Used",
    "account_n_merchants":        "Unique Merchants Used",
    "account_n_locations":        "Unique Locations",
    "account_avg_login_attempts": "Avg. Login Attempts (Account)",
    "amount_to_balance_ratio":    "Amount-to-Balance Ratio",
    "amount_z_score":             "Amount Z-Score (vs Account)",
    "is_large_amount":            "Unusually Large Amount",
    "is_long_duration":           "Unusually Long Duration",
    "is_low_balance":             "Low Account Balance",
    "is_high_login_attempts":     "High Login Attempts Flag",
    "is_multi_device":            "Multi-Device Account Flag",
    "TransactionType_enc":        "Transaction Type",
    "Channel_enc":                "Channel",
    "CustomerOccupation_enc":     "Customer Occupation",
}
