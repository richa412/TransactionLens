"""
data_preprocessing.py
---------------------
Loads the raw CSV, cleans it, and returns a tidy DataFrame ready for
feature engineering.  Every transformation step is documented so a
beginner data analyst can follow along.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column names exactly as they appear in the CSV header
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "PreviousTransactionDate",
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Read the CSV and return a raw DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {filepath}")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary that describes the dataset:
    shape, dtypes, missing values, duplicates, unique counts.
    Useful for the EDA section of the Streamlit app.
    """
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
        "unique_counts": df.nunique().to_dict(),
    }
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps:
    1. Remove duplicate rows.
    2. Parse datetime columns.
    3. Coerce numeric columns to the correct type.
    4. Strip whitespace from string columns.
    5. Rename 'IP Address' → 'IPAddress' for easier attribute access.
    """
    df = df.copy()

    # -- 1. Duplicates --
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info(f"Removed {dropped} duplicate rows.")

    # -- 2. Rename tricky column --
    df = df.rename(columns={"IP Address": "IPAddress"})

    # -- 3. Parse datetime columns --
    for col in ["TransactionDate", "PreviousTransactionDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info("Parsed TransactionDate and PreviousTransactionDate as datetime.")

    # -- 4. Numeric coercion --
    for col in ["TransactionAmount", "AccountBalance", "TransactionDuration",
                "LoginAttempts", "CustomerAge"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- 5. Strip whitespace from object columns --
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # -- 6. Drop rows where critical fields are completely null --
    critical = ["TransactionAmount", "TransactionDate", "AccountID"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with null critical fields.")

    # -- 7. Fill remaining small numeric gaps with column median --
    num_cols = ["TransactionDuration", "LoginAttempts", "CustomerAge", "AccountBalance"]
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing {col} with median={median_val:.2f}")

    logger.info(f"Clean dataset: {len(df):,} rows remain.")
    return df


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Return columns that are object/string type (i.e., categorical)."""
    return list(df.select_dtypes(include="object").columns)


def get_numerical_columns(df: pd.DataFrame) -> list:
    """Return columns that are numeric type."""
    return list(df.select_dtypes(include=[np.number]).columns)


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data → {out_path}")


# ---------------------------------------------------------------------------
# Standalone execution: run this file directly to clean the raw CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RAW_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "bank_transactions_data_2.csv"
    )
    OUT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "transactions_clean.csv"
    )

    raw_df = load_raw_data(RAW_PATH)

    summary = inspect_data(raw_df)
    print("\n=== Dataset Inspection ===")
    print(f"  Rows       : {summary['n_rows']:,}")
    print(f"  Columns    : {summary['n_cols']}")
    print(f"  Duplicates : {summary['n_duplicates']}")
    print(f"\nMissing values:\n{pd.Series(summary['missing_values'])}")
    print(f"\nUnique counts:\n{pd.Series(summary['unique_counts'])}")
    print(f"\nData types:\n{pd.Series(summary['dtypes'])}")

    clean_df = clean_data(raw_df)
    save_processed(clean_df, OUT_PATH)
    print(f"\nProcessed file saved to: {OUT_PATH}")
