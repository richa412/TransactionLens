"""
feature_engineering.py
-----------------------
Derives new features from the cleaned transaction DataFrame.
Only features that can be computed from the actual columns are created:
    TransactionID, AccountID, TransactionAmount, TransactionDate,
    TransactionType, Location, DeviceID, IPAddress, MerchantID, Channel,
    CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts,
    AccountBalance, PreviousTransactionDate
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, weekday from TransactionDate."""
    df = df.copy()
    df["tx_hour"]       = df["TransactionDate"].dt.hour
    df["tx_day"]        = df["TransactionDate"].dt.day
    df["tx_month"]      = df["TransactionDate"].dt.month
    df["tx_weekday"]    = df["TransactionDate"].dt.dayofweek   # 0=Mon, 6=Sun
    df["tx_is_weekend"] = (df["tx_weekday"] >= 5).astype(int)

    # Time (in hours) since the previous transaction on the same account
    # PreviousTransactionDate is stored in the dataset as a raw column.
    df["hours_since_prev_tx"] = (
        (df["TransactionDate"] - df["PreviousTransactionDate"])
        .dt.total_seconds()
        / 3600
    )
    # Negative values would mean the previous date is *after* the transaction date —
    # treat them as NaN (data issue) and fill with the median.
    df["hours_since_prev_tx"] = df["hours_since_prev_tx"].clip(lower=0)
    median_hrs = df["hours_since_prev_tx"].median()
    df["hours_since_prev_tx"] = df["hours_since_prev_tx"].fillna(median_hrs)

    logger.info("Added temporal features.")
    return df


# ---------------------------------------------------------------------------
# Account-level aggregation features
# ---------------------------------------------------------------------------

def add_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AccountID compute:
      - account_tx_count          : total number of transactions
      - account_total_amount      : total transaction amount
      - account_avg_amount        : mean transaction amount
      - account_std_amount        : std of transaction amount
      - account_n_devices         : number of distinct devices used
      - account_n_merchants       : number of distinct merchants
      - account_n_locations       : number of distinct locations
      - account_avg_login_attempts: mean login attempts
    """
    df = df.copy()

    grp = df.groupby("AccountID")

    agg = grp.agg(
        account_tx_count          = ("TransactionAmount", "count"),
        account_total_amount      = ("TransactionAmount", "sum"),
        account_avg_amount        = ("TransactionAmount", "mean"),
        account_std_amount        = ("TransactionAmount", "std"),
        account_n_devices         = ("DeviceID",          "nunique"),
        account_n_merchants       = ("MerchantID",        "nunique"),
        account_n_locations       = ("Location",          "nunique"),
        account_avg_login_attempts= ("LoginAttempts",     "mean"),
    ).reset_index()

    # std can be NaN for accounts with a single transaction — fill with 0
    agg["account_std_amount"] = agg["account_std_amount"].fillna(0)

    df = df.merge(agg, on="AccountID", how="left")
    logger.info("Added account-level aggregation features.")
    return df


# ---------------------------------------------------------------------------
# Transaction-level risk indicators
# ---------------------------------------------------------------------------

def add_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-transaction signals that may indicate anomalous behaviour.
      - amount_to_balance_ratio : TransactionAmount / (AccountBalance + 1)
      - amount_vs_account_avg   : how many std-devs the amount is from the
                                   account's own mean  (z-score style)
      - is_high_login_attempts  : LoginAttempts > 3  (boolean flag)
      - is_long_duration        : TransactionDuration > global 95th pct
      - is_large_amount         : TransactionAmount  > global 95th pct
      - is_low_balance          : AccountBalance     < global 5th pct
    """
    df = df.copy()

    # Ratio of transaction to balance (high ratio = spending most/all funds)
    df["amount_to_balance_ratio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)

    # Per-account z-score for transaction amount
    safe_std = df["account_std_amount"].replace(0, np.nan)
    df["amount_z_score"] = (
        (df["TransactionAmount"] - df["account_avg_amount"]) / safe_std
    ).fillna(0)

    # Binary flags based on global percentiles
    p95_amount   = df["TransactionAmount"].quantile(0.95)
    p95_duration = df["TransactionDuration"].quantile(0.95)
    p05_balance  = df["AccountBalance"].quantile(0.05)

    df["is_large_amount"]        = (df["TransactionAmount"]   > p95_amount  ).astype(int)
    df["is_long_duration"]       = (df["TransactionDuration"] > p95_duration).astype(int)
    df["is_low_balance"]         = (df["AccountBalance"]      < p05_balance ).astype(int)
    df["is_high_login_attempts"] = (df["LoginAttempts"]       > 3           ).astype(int)

    # Flag if this account uses an unusually high number of devices
    p90_devices = df["account_n_devices"].quantile(0.90)
    df["is_multi_device"] = (df["account_n_devices"] > p90_devices).astype(int)

    logger.info("Added per-transaction risk indicator features.")
    return df


# ---------------------------------------------------------------------------
# Age group feature
# ---------------------------------------------------------------------------

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin CustomerAge into readable groups."""
    df = df.copy()
    bins   = [0, 25, 35, 50, 65, 120]
    labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
    df["age_group"] = pd.cut(df["CustomerAge"], bins=bins, labels=labels, right=True)
    df["age_group"] = df["age_group"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Encode categoricals for model input
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode low-cardinality string columns needed by the ML model.
    Returns the DataFrame with added *_enc columns.
    Keeps the original columns intact for display purposes.
    """
    df = df.copy()
    for col in ["TransactionType", "Channel", "CustomerOccupation"]:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# Master pipeline — call this to get the fully engineered DataFrame
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in order."""
    df = add_temporal_features(df)
    df = add_account_features(df)
    df = add_risk_indicators(df)
    df = add_age_group(df)
    df = encode_categoricals(df)
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# List the feature columns used as model input
# ---------------------------------------------------------------------------

MODEL_FEATURES = [
    # Transaction-level numeric
    "TransactionAmount",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "CustomerAge",
    # Temporal
    "tx_hour",
    "tx_month",
    "tx_weekday",
    "tx_is_weekend",
    "hours_since_prev_tx",
    # Account aggregates
    "account_tx_count",
    "account_avg_amount",
    "account_std_amount",
    "account_n_devices",
    "account_n_merchants",
    "account_n_locations",
    "account_avg_login_attempts",
    # Risk indicators
    "amount_to_balance_ratio",
    "amount_z_score",
    "is_large_amount",
    "is_long_duration",
    "is_low_balance",
    "is_high_login_attempts",
    "is_multi_device",
    # Encoded categoricals
    "TransactionType_enc",
    "Channel_enc",
    "CustomerOccupation_enc",
]
