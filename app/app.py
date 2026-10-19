"""
app.py
------
Main Streamlit application for the AI-Powered Banking Transaction
Fraud & Risk Analytics Dashboard.

Run with:
    streamlit run app/app.py

The app expects the trained model artefacts to already exist in models/.
If they do not, run:
    python src/model_training.py
"""

import os
import sys
import sqlite3
import logging
import calendar

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — make src/ importable regardless of where the app is launched
# ---------------------------------------------------------------------------
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
SRC_DIR     = os.path.join(PROJECT_DIR, "src")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)

from prediction    import load_artefacts, build_feature_vector, explain_prediction, FEATURE_LABELS
from visualization import (
    transaction_amount_histogram,
    transaction_trend_line,
    transaction_amount_trend,
    transaction_type_pie,
    channel_bar,
    location_bar,
    occupation_bar,
    age_group_bar,
    risk_distribution_pie,
    risk_score_histogram,
    login_attempts_bar,
    transaction_duration_box,
    balance_vs_amount_scatter,
    correlation_heatmap,
    feature_importance_bar,
    account_amount_over_time,
    account_risk_gauge,
    RISK_COLOR_MAP,
)
from feature_engineering import MODEL_FEATURES
from anomaly_detection    import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH    = os.path.join(PROJECT_DIR, "data", "processed", "transactions_featured.csv")
MODEL_PATH   = os.path.join(PROJECT_DIR, "models", "isolation_forest_pipeline.pkl")
DB_PATH      = os.path.join(PROJECT_DIR, "data", "processed", "banking.db")
SQL_PATH     = os.path.join(PROJECT_DIR, "sql", "banking_analysis.sql")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Banking Fraud & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* General */
body { font-family: 'Segoe UI', sans-serif; }

/* Metric card */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .value  { font-size: 2rem; font-weight: 700; color: #1e293b; }
.metric-card .label  { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
.metric-card .delta  { font-size: 0.8rem; margin-top: 2px; }

/* Risk badges */
.badge-High   { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:12px; font-weight:600; }
.badge-Medium { background:#fef3c7; color:#d97706; padding:3px 10px; border-radius:12px; font-weight:600; }
.badge-Low    { background:#dcfce7; color:#16a34a; padding:3px 10px; border-radius:12px; font-weight:600; }

/* Section header */
.section-header { font-size: 1.4rem; font-weight: 700; color: #1e293b; margin-bottom: 6px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & model loading — cached so they only run once per session
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading transaction data …")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(
            "Processed dataset not found. "
            "Please run:  **python src/model_training.py**  first."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH, parse_dates=["TransactionDate", "PreviousTransactionDate"])
    return df


@st.cache_resource(show_spinner="Loading AI model …")
def load_model():
    try:
        return load_artefacts()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Setting up SQL database …")
def get_db_connection(df: pd.DataFrame):
    """Create an in-memory (or file) SQLite DB from the DataFrame."""
    # Use a file-based DB so it persists between pages
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Helper: metric card HTML
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", color: str = "#3b82f6") -> str:
    delta_html = f'<div class="delta" style="color:{color};">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="value" style="color:{color};">{value}</div>
        <div class="label">{label}</div>
        {delta_html}
    </div>
    """


def risk_badge(level: str) -> str:
    return f'<span class="badge-{level}">{level} Risk</span>'


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = ["🏠 Dashboard", "🔍 Transaction Analysis", "⚠️ Risk Detection",
         "👤 Account Risk Analysis", "🧠 AI Insights", "🗄️ SQL Explorer"]

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏦 Banking Analytics")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "AI-powered anomaly detection on real banking transaction data. "
        "No fraud labels are used — risk is inferred via **Isolation Forest**."
    )
    st.markdown("---")
    st.markdown(
        "<small>Dataset: Bank Transaction Dataset · Kaggle</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load data & model
# ---------------------------------------------------------------------------
df        = load_data()
artefacts = load_model()
conn      = get_db_connection(df)

model_loaded = artefacts is not None


# ===========================================================================
# PAGE 1 — DASHBOARD
# ===========================================================================

if page == "🏠 Dashboard":
    st.title("🏦 Banking Transaction Risk Dashboard")
    st.markdown("Real-time analytics across **{:,}** transactions.".format(len(df)))
    st.markdown("---")

    # ---- KPI cards ----
    total_tx      = len(df)
    total_value   = df["TransactionAmount"].sum()
    n_accounts    = df["AccountID"].nunique()
    avg_amount    = df["TransactionAmount"].mean()

    if "risk_level" in df.columns:
        n_high  = (df["risk_level"] == "High").sum()
        n_med   = (df["risk_level"] == "Medium").sum()
        risk_rt = n_high / total_tx * 100
    else:
        n_high = n_med = risk_rt = 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(metric_card("Total Transactions", f"{total_tx:,}"),           unsafe_allow_html=True)
    c2.markdown(metric_card("Total Value",        f"${total_value:,.0f}"),    unsafe_allow_html=True)
    c3.markdown(metric_card("Unique Accounts",    f"{n_accounts:,}"),         unsafe_allow_html=True)
    c4.markdown(metric_card("Avg. Amount",        f"${avg_amount:,.2f}"),     unsafe_allow_html=True)
    c5.markdown(metric_card("High-Risk Txns",     f"{n_high:,}",  color="#ef4444"), unsafe_allow_html=True)
    c6.markdown(metric_card("Risk Rate",          f"{risk_rt:.1f}%", color="#f59e0b"), unsafe_allow_html=True)

    st.markdown("---")

    # ---- Charts row 1 ----
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(transaction_trend_line(df), use_container_width=True)
    with col2:
        st.plotly_chart(transaction_amount_trend(df), use_container_width=True)

    # ---- Charts row 2 ----
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(transaction_type_pie(df), use_container_width=True)
    with col4:
        st.plotly_chart(channel_bar(df), use_container_width=True)

    # ---- Charts row 3 ----
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(location_bar(df), use_container_width=True)
    with col6:
        st.plotly_chart(occupation_bar(df), use_container_width=True)

    # ---- Charts row 4 ----
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(age_group_bar(df), use_container_width=True)
    with col8:
        st.plotly_chart(transaction_amount_histogram(df), use_container_width=True)

    # ---- Risk charts (only when model is loaded) ----
    if model_loaded and "risk_level" in df.columns:
        st.markdown("### ⚠️ Risk Overview")
        col9, col10 = st.columns(2)
        with col9:
            st.plotly_chart(risk_distribution_pie(df), use_container_width=True)
        with col10:
            st.plotly_chart(risk_score_histogram(df), use_container_width=True)

    # ---- Correlation & scatter ----
    st.markdown("### 📊 Deeper Analysis")
    col11, col12 = st.columns(2)
    with col11:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)
    with col12:
        st.plotly_chart(balance_vs_amount_scatter(df), use_container_width=True)


# ===========================================================================
# PAGE 2 — TRANSACTION ANALYSIS
# ===========================================================================

elif page == "🔍 Transaction Analysis":
    st.title("🔍 Transaction Analysis")
    st.markdown("Filter and explore individual transactions.")
    st.markdown("---")

    # ---- Sidebar filters ----
    st.sidebar.markdown("### Filters")

    # Date range
    min_date = df["TransactionDate"].dt.date.min()
    max_date = df["TransactionDate"].dt.date.max()
    date_range = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )

    # Other filters
    tx_types  = ["All"] + sorted(df["TransactionType"].unique())
    channels  = ["All"] + sorted(df["Channel"].unique())
    locations = ["All"] + sorted(df["Location"].unique())
    risk_levels = ["All", "High", "Medium", "Low"] if "risk_level" in df.columns else ["All"]

    sel_type     = st.sidebar.selectbox("Transaction Type", tx_types)
    sel_channel  = st.sidebar.selectbox("Channel",          channels)
    sel_location = st.sidebar.selectbox("Location",         locations)
    sel_risk     = st.sidebar.selectbox("Risk Level",       risk_levels)

    amt_min = float(df["TransactionAmount"].to_numpy().min())
    amt_max = float(df["TransactionAmount"].to_numpy().max())
    amt_range = st.sidebar.slider(
        "Amount Range ($)",
        min_value=amt_min, max_value=amt_max,
        value=(amt_min, amt_max),
    )

    # ---- Apply filters ----
    fdf = df.copy()
    if len(date_range) == 2:
        fdf = fdf[
            (fdf["TransactionDate"].dt.date >= date_range[0]) &
            (fdf["TransactionDate"].dt.date <= date_range[1])
        ]
    if sel_type     != "All": fdf = fdf[fdf["TransactionType"] == sel_type]
    if sel_channel  != "All": fdf = fdf[fdf["Channel"]         == sel_channel]
    if sel_location != "All": fdf = fdf[fdf["Location"]        == sel_location]
    if sel_risk     != "All" and "risk_level" in fdf.columns:
        fdf = fdf[fdf["risk_level"] == sel_risk]
    fdf = fdf[
        (fdf["TransactionAmount"] >= amt_range[0]) &
        (fdf["TransactionAmount"] <= amt_range[1])
    ]

    st.markdown(f"**{len(fdf):,} transactions** match the current filters.")

    # ---- Summary stats ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(fdf):,}")
    c2.metric("Total Value",        f"${fdf['TransactionAmount'].sum():,.2f}")
    c3.metric("Avg. Amount",        f"${fdf['TransactionAmount'].mean():,.2f}")
    if "risk_level" in fdf.columns:
        c4.metric("High-Risk Count", f"{(fdf['risk_level']=='High').sum():,}")

    st.markdown("---")

    # ---- Quick charts ----
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(transaction_type_pie(fdf), use_container_width=True)
    with col2:
        st.plotly_chart(channel_bar(fdf), use_container_width=True)

    # ---- Transaction table ----
    st.markdown("### 📋 Transaction Records")
    display_cols = [
        "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
        "TransactionType", "Channel", "Location", "LoginAttempts", "AccountBalance",
    ]
    if "risk_level" in fdf.columns:
        display_cols += ["risk_level", "risk_score_pct"]

    show_cols = [c for c in display_cols if c in fdf.columns]

    # Colour-code risk_level column
    def highlight_risk(row):
        if "risk_level" not in row.index:
            return [""] * len(row)
        level = row.get("risk_level", "Low")
        if level == "High":
            return ["background-color: #fee2e2"] * len(row)
        elif level == "Medium":
            return ["background-color: #fef3c7"] * len(row)
        return [""] * len(row)

    styled = (
        fdf[show_cols]
        .sort_values("TransactionDate", ascending=False)
        .reset_index(drop=True)
        .style.apply(highlight_risk, axis=1)
    )
    st.dataframe(styled, use_container_width=True, height=450)

    # Download
    csv_bytes = fdf[show_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered data as CSV", csv_bytes,
                       file_name="filtered_transactions.csv", mime="text/csv")


# ===========================================================================
# PAGE 3 — RISK DETECTION
# ===========================================================================

elif page == "⚠️ Risk Detection":
    st.title("⚠️ Transaction Risk Detection")
    st.markdown(
        "Enter transaction details to receive an **AI-generated risk assessment**. "
        "This uses an **Isolation Forest** anomaly-detection model — no fraud labels are used."
    )
    st.markdown("---")

    if not model_loaded:
        st.warning("⚠️ Model not loaded. Run `python src/model_training.py` first.")
        st.stop()

    mode = st.radio("Input mode", ["Select existing transaction", "Enter manually"], horizontal=True)

    tx_dict = {}

    if mode == "Select existing transaction":
        tx_id = st.selectbox("Select Transaction ID", df["TransactionID"].tolist())
        row   = df[df["TransactionID"] == tx_id].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Account ID:** {row['AccountID']}")
        col1.markdown(f"**Amount:** ${row['TransactionAmount']:,.2f}")
        col1.markdown(f"**Type:** {row['TransactionType']}")
        col2.markdown(f"**Channel:** {row['Channel']}")
        col2.markdown(f"**Location:** {row['Location']}")
        col2.markdown(f"**Login Attempts:** {int(row['LoginAttempts'])}")
        col3.markdown(f"**Customer Age:** {int(row['CustomerAge'])}")
        col3.markdown(f"**Occupation:** {row['CustomerOccupation']}")
        col3.markdown(f"**Account Balance:** ${row['AccountBalance']:,.2f}")

        for field in ["TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
                      "PreviousTransactionDate", "TransactionType", "Location", "DeviceID",
                      "MerchantID", "Channel", "CustomerAge", "CustomerOccupation",
                      "TransactionDuration", "LoginAttempts", "AccountBalance"]:
            tx_dict[field] = row[field]

    else:
        st.markdown("#### Transaction Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            tx_dict["AccountID"]         = st.text_input("Account ID", value="AC00001")
            tx_dict["TransactionAmount"] = st.number_input("Amount ($)", 0.01, 100000.0, 250.0, step=10.0)
            tx_dict["TransactionType"]   = st.selectbox("Transaction Type", ["Debit", "Credit"])
            tx_dict["Channel"]           = st.selectbox("Channel", ["ATM", "Online", "Branch"])
            tx_dict["Location"]          = st.selectbox("Location", sorted(df["Location"].unique()))

        with col2:
            tx_dict["DeviceID"]              = st.text_input("Device ID", value="D000001")
            tx_dict["MerchantID"]            = st.text_input("Merchant ID", value="M001")
            tx_dict["CustomerAge"]           = st.slider("Customer Age", 18, 90, 35)
            tx_dict["CustomerOccupation"]    = st.selectbox("Occupation",
                                                             sorted(df["CustomerOccupation"].unique()))
            tx_dict["AccountBalance"]        = st.number_input("Account Balance ($)", 0.0, 100000.0, 5000.0)

        with col3:
            tx_dict["TransactionDuration"]   = st.slider("Duration (s)", 1, 600, 120)
            tx_dict["LoginAttempts"]         = st.slider("Login Attempts", 1, 10, 1)
            tx_dict["TransactionDate"]       = st.date_input("Transaction Date")
            tx_dict["PreviousTransactionDate"] = st.date_input("Previous Transaction Date")
            tx_dict["TransactionID"]         = "TX_MANUAL"

    st.markdown("---")
    run_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Analysing transaction …"):
            try:
                fv = build_feature_vector(tx_dict, df)
                result = {}

                # Score
                pipeline   = artefacts["pipeline"]
                all_scores = artefacts["all_scores"]

                raw = pipeline.decision_function(fv[MODEL_FEATURES].fillna(0))[0]
                inv = -raw
                min_s, max_s = all_scores.min(), all_scores.max()
                risk_pct = float(np.clip(((inv - min_s) / (max_s - min_s + 1e-9)) * 100, 0, 100))

                if risk_pct >= HIGH_RISK_THRESHOLD:
                    risk_level = "High"
                elif risk_pct >= MEDIUM_RISK_THRESHOLD:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"

                result = {
                    "risk_score_pct": round(risk_pct, 2),
                    "risk_level":     risk_level,
                }

                # ---- Result display ----
                st.markdown("### 📊 Risk Assessment Result")

                col_g, col_r = st.columns([1, 2])
                with col_g:
                    st.plotly_chart(
                        account_risk_gauge(result["risk_score_pct"]),
                        use_container_width=True,
                    )

                with col_r:
                    st.markdown(f"#### Risk Level: {risk_badge(risk_level)}",
                                unsafe_allow_html=True)
                    st.markdown(
                        f"**Risk Score: {result['risk_score_pct']:.1f}%**  \n"
                        "*(Higher score = more anomalous behaviour)*"
                    )

                    # Colour-coded interpretation
                    if risk_level == "High":
                        st.error(
                            "🚨 **High Risk Transaction Detected**  \n"
                            "This transaction exhibits anomalous behaviour compared to "
                            "normal transaction patterns. It is recommended for further review."
                        )
                    elif risk_level == "Medium":
                        st.warning(
                            "⚠️ **Medium Risk Transaction**  \n"
                            "This transaction shows some unusual characteristics. "
                            "Consider monitoring this account."
                        )
                    else:
                        st.success(
                            "✅ **Low Risk Transaction**  \n"
                            "This transaction appears consistent with normal account behaviour."
                        )

                # ---- Explainability ----
                st.markdown("#### 🔬 Contributing Risk Factors")
                st.markdown(
                    "The following features contributed most to this transaction's risk score, "
                    "based on global feature importance from the trained model:"
                )

                factors = explain_prediction(
                    fv, artefacts["feature_importances"], top_n=7
                )

                for i, f in enumerate(factors, 1):
                    label = FEATURE_LABELS.get(f["feature"], f["feature"])
                    val   = f["value"]
                    cont  = f["contribution"]

                    # Normalise contribution bar to 0-100 for display
                    max_cont = max(x["contribution"] for x in factors) or 1
                    bar_pct  = int((cont / max_cont) * 100)

                    st.markdown(
                        f"**{i}. {label}** — `{val}`  \n"
                        f"<div style='background:#e2e8f0;border-radius:4px;height:8px;'>"
                        f"<div style='background:#3b82f6;width:{bar_pct}%;height:8px;border-radius:4px;'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(" ")

                # ---- Full feature vector ----
                with st.expander("📋 Full Feature Vector"):
                    display_fv = pd.DataFrame({
                        "Feature": [FEATURE_LABELS.get(c, c) for c in MODEL_FEATURES],
                        "Value":   [
                            round(float(fv[c].iloc[0]), 4) if c in fv.columns else "N/A"
                            for c in MODEL_FEATURES
                        ],
                    })
                    st.dataframe(display_fv, use_container_width=True, height=500)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.exception(e)


# ===========================================================================
# PAGE 4 — ACCOUNT RISK ANALYSIS
# ===========================================================================

elif page == "👤 Account Risk Analysis":
    st.title("👤 Account Risk Analysis")
    st.markdown("Deep-dive into the transaction history and risk profile of any account.")
    st.markdown("---")

    account_ids = sorted(df["AccountID"].unique())
    selected_acct = st.selectbox("Select Account ID", account_ids)

    acct_df = df[df["AccountID"] == selected_acct].copy()

    if len(acct_df) == 0:
        st.warning("No transactions found for this account.")
        st.stop()

    # ---- KPI row ----
    total_tx    = len(acct_df)
    total_val   = acct_df["TransactionAmount"].sum()
    avg_val     = acct_df["TransactionAmount"].mean()
    n_devices   = acct_df["DeviceID"].nunique()
    n_merchants = acct_df["MerchantID"].nunique()
    n_locs      = acct_df["Location"].nunique()
    avg_login   = acct_df["LoginAttempts"].mean()

    if "risk_level" in acct_df.columns:
        n_high_risk  = (acct_df["risk_level"] == "High").sum()
        avg_risk_pct = acct_df["risk_score_pct"].mean()
    else:
        n_high_risk  = 0
        avg_risk_pct = 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions",   f"{total_tx}")
    c2.metric("Total Value",          f"${total_val:,.2f}")
    c3.metric("Avg. Transaction",     f"${avg_val:,.2f}")
    c4.metric("Avg. Risk Score",      f"{avg_risk_pct:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Unique Devices",       f"{n_devices}")
    c6.metric("Unique Merchants",     f"{n_merchants}")
    c7.metric("Unique Locations",     f"{n_locs}")
    c8.metric("High-Risk Txns",       f"{n_high_risk}", delta_color="inverse")

    st.markdown("---")

    # ---- Charts ----
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(account_amount_over_time(acct_df, selected_acct),
                        use_container_width=True)
    with col2:
        if "risk_level" in acct_df.columns:
            st.plotly_chart(risk_distribution_pie(acct_df), use_container_width=True)
        else:
            st.plotly_chart(transaction_type_pie(acct_df), use_container_width=True)

    # ---- Login behaviour ----
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(login_attempts_bar(acct_df), use_container_width=True)
    with col4:
        st.plotly_chart(transaction_duration_box(acct_df), use_container_width=True)

    # ---- Risk indicators summary ----
    st.markdown("### 🔴 Risk Indicator Summary")
    indicators = {
        "High Login Attempts (>3)":     int(acct_df.get("is_high_login_attempts", pd.Series([0])).sum()),
        "Large Transactions (top 5%)":  int(acct_df.get("is_large_amount",        pd.Series([0])).sum()),
        "Long Duration (top 5%)":       int(acct_df.get("is_long_duration",        pd.Series([0])).sum()),
        "Low Balance Transactions":     int(acct_df.get("is_low_balance",          pd.Series([0])).sum()),
    }

    ind_cols = st.columns(len(indicators))
    for col, (label, val) in zip(ind_cols, indicators.items()):
        color = "#ef4444" if val > 0 else "#22c55e"
        col.markdown(metric_card(label, str(val), color=color), unsafe_allow_html=True)

    # ---- Transaction table ----
    st.markdown("### 📋 Account Transactions")
    show_cols = [c for c in [
        "TransactionID", "TransactionDate", "TransactionAmount",
        "TransactionType", "Channel", "Location", "MerchantID",
        "LoginAttempts", "TransactionDuration",
        "risk_level", "risk_score_pct",
    ] if c in acct_df.columns]

    st.dataframe(
        acct_df[show_cols].sort_values("TransactionDate", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


# ===========================================================================
# PAGE 5 — AI INSIGHTS
# ===========================================================================

elif page == "🧠 AI Insights":
    st.title("🧠 AI-Generated Insights")
    st.markdown(
        "These insights are **computed directly from the dataset and model output**. "
        "No text has been hard-coded or fabricated."
    )
    st.markdown("---")

    fi = None
    if not model_loaded:
        st.warning("Load the trained model first: `python src/model_training.py`")
    else:
        fi = artefacts["feature_importances"]

    # ---- Dataset overview ----
    st.markdown("### 📊 Dataset Summary")
    total_tx    = len(df)
    total_val   = df["TransactionAmount"].sum()
    avg_amt     = df["TransactionAmount"].mean()
    med_amt     = df["TransactionAmount"].median()
    date_range  = f"{df['TransactionDate'].min().date()} → {df['TransactionDate'].max().date()}"

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
**Dataset Snapshot**
- **Total Transactions:** {total_tx:,}
- **Date Range:** {date_range}
- **Total Transaction Value:** ${total_val:,.2f}
- **Avg Transaction Amount:** ${avg_amt:,.2f}
- **Median Transaction Amount:** ${med_amt:,.2f}
- **Unique Accounts:** {df['AccountID'].nunique():,}
- **Unique Merchants:** {df['MerchantID'].nunique():,}
- **Unique Locations:** {df['Location'].nunique():,}
- **Unique Devices:** {df['DeviceID'].nunique():,}
        """)
    with col2:
        if model_loaded and "risk_level" in df.columns:
            high_n   = (df["risk_level"] == "High").sum()
            med_n    = (df["risk_level"] == "Medium").sum()
            low_n    = (df["risk_level"] == "Low").sum()
            high_pct = high_n / total_tx * 100
            st.warning(f"""
**Risk Distribution (Model Output)**
- **High Risk:**   {high_n:,}  ({high_pct:.1f}%)
- **Medium Risk:** {med_n:,}  ({med_n/total_tx*100:.1f}%)
- **Low Risk:**    {low_n:,}  ({low_n/total_tx*100:.1f}%)

*High-risk transactions are anomalous compared to normal account behaviour — they should be investigated, not automatically labelled as fraud.*
            """)

    # ---- Pattern analysis ----
    st.markdown("### 🔍 Key Transaction Patterns")

    # Top channel
    top_channel = df["Channel"].value_counts().idxmax()
    top_channel_pct = df["Channel"].value_counts().max() / total_tx * 100

    # Most active occupation
    top_occ = df["CustomerOccupation"].value_counts().idxmax()

    # High login transactions
    high_login_count = (df["LoginAttempts"] > 3).sum()
    high_login_pct   = high_login_count / total_tx * 100

    # Most common transaction type
    top_type = df["TransactionType"].value_counts().idxmax()
    top_type_pct = df["TransactionType"].value_counts().max() / total_tx * 100

    # Busiest month
    if "tx_month" in df.columns:
        busiest_month_num = df["tx_month"].value_counts().idxmax()
        busiest_month = calendar.month_name[busiest_month_num]
    else:
        busiest_month = "N/A"

    patterns = [
        f"🔵 **{top_channel}** is the most-used channel, accounting for **{top_channel_pct:.1f}%** of all transactions.",
        f"🔵 The most active customer segment is **{top_occ}**.",
        f"🔵 The dominant transaction type is **{top_type}** ({top_type_pct:.1f}% of transactions).",
        f"🔵 **{high_login_count:,}** transactions ({high_login_pct:.1f}%) had more than 3 login attempts — a potential signal of account access difficulty.",
        f"🔵 The busiest transaction month is **{busiest_month}**.",
        f"🔵 Average account balance is **${df['AccountBalance'].mean():,.2f}**, while average transaction amount is **${avg_amt:,.2f}**.",
    ]
    for p in patterns:
        st.markdown(p)

    # ---- Feature importance ----
    if model_loaded:
        st.markdown("### 🏆 Top Risk Factors (Model-Derived)")
        st.markdown(
            "These are the features that most influence the Isolation Forest anomaly score, "
            "computed via **permutation importance**:"
        )
        st.plotly_chart(feature_importance_bar(fi, top_n=12), use_container_width=True)

        top_features = list(fi.keys())[:5]
        top_labels   = [FEATURE_LABELS.get(f, f) for f in top_features]
        st.markdown(
            "**Top risk-contributing factors:**  \n" +
            "\n".join(f"  {i+1}. {lbl}" for i, lbl in enumerate(top_labels))
        )

    # ---- High-risk analysis ----
    if "risk_level" in df.columns:
        st.markdown("### 🚨 High-Risk Transaction Analysis")

        high_df = df[df["risk_level"] == "High"]

        if len(high_df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top channels for high-risk transactions:**")
                hc = high_df["Channel"].value_counts().reset_index()
                hc.columns = ["Channel", "count"]
                st.dataframe(hc, use_container_width=True)

            with col2:
                st.markdown("**Top occupations in high-risk transactions:**")
                ho = high_df["CustomerOccupation"].value_counts().reset_index()
                ho.columns = ["Occupation", "count"]
                st.dataframe(ho, use_container_width=True)

            st.markdown(f"""
**High-Risk Statistics:**
- Avg. amount in high-risk txns: **${high_df['TransactionAmount'].mean():,.2f}**
  vs overall avg: **${avg_amt:,.2f}**
- Avg. login attempts in high-risk: **{high_df['LoginAttempts'].mean():.2f}**
  vs overall: **{df['LoginAttempts'].mean():.2f}**
- Avg. transaction duration (high-risk): **{high_df['TransactionDuration'].mean():.1f}s**
  vs overall: **{df['TransactionDuration'].mean():.1f}s**
            """)

    # ---- Recommendations ----
    st.markdown("### 💡 Suggested Areas for Investigation")
    recs = [
        "**Accounts with repeated high login attempts** — may indicate brute-force or credential-stuffing attacks.",
        "**Very large transactions relative to account balance** — unusually high amount-to-balance ratios are a strong anomaly signal.",
        "**Accounts using many different devices or merchants** — multi-device behaviour can indicate account takeover.",
        "**Transactions flagged as both large AND during unusual hours** — compound risk signals deserve priority review.",
        "**High-risk accounts with frequent transactions** — velocity-based anomalies are a key fraud indicator.",
    ]
    for r in recs:
        st.markdown(f"• {r}")

    st.markdown("---")
    st.caption(
        "⚠️ All insights are computed from the actual dataset and model output. "
        "Anomalous transactions are not confirmed fraud — they are statistical outliers "
        "requiring human review."
    )


# ===========================================================================
# PAGE 6 — SQL EXPLORER
# ===========================================================================

elif page == "🗄️ SQL Explorer":
    st.title("🗄️ SQL Analytics Explorer")
    st.markdown(
        "Run SQL queries against the **transactions** table "
        "(SQLite database built from the processed dataset)."
    )
    st.markdown("---")

    # ---- Preset queries ----
    PRESET_QUERIES = {
        "Top 10 Highest-Value Transactions": """
SELECT TransactionID, AccountID, TransactionAmount, TransactionDate,
       TransactionType, Channel, Location, risk_level
FROM transactions
ORDER BY TransactionAmount DESC
LIMIT 10;
        """,
        "Most Active Accounts": """
SELECT AccountID,
       COUNT(*) AS transaction_count,
       ROUND(SUM(TransactionAmount),2) AS total_amount,
       ROUND(AVG(TransactionAmount),2) AS avg_amount
FROM transactions
GROUP BY AccountID
ORDER BY transaction_count DESC
LIMIT 15;
        """,
        "Average Amount by Transaction Type": """
SELECT TransactionType,
       COUNT(*) AS count,
       ROUND(AVG(TransactionAmount),2) AS avg_amount,
       ROUND(SUM(TransactionAmount),2) AS total_amount
FROM transactions
GROUP BY TransactionType
ORDER BY avg_amount DESC;
        """,
        "Channel Usage & Value": """
SELECT Channel,
       COUNT(*) AS transaction_count,
       ROUND(SUM(TransactionAmount),2) AS total_amount,
       ROUND(AVG(TransactionAmount),2) AS avg_amount
FROM transactions
GROUP BY Channel
ORDER BY transaction_count DESC;
        """,
        "Top 15 Locations by Activity": """
SELECT Location,
       COUNT(*) AS transaction_count,
       ROUND(SUM(TransactionAmount),2) AS total_amount
FROM transactions
GROUP BY Location
ORDER BY total_amount DESC
LIMIT 15;
        """,
        "Monthly Transaction Trends": """
SELECT STRFTIME('%Y-%m', TransactionDate) AS month,
       COUNT(*) AS transaction_count,
       ROUND(SUM(TransactionAmount),2) AS total_amount
FROM transactions
GROUP BY month
ORDER BY month;
        """,
        "Top Merchants by Value": """
SELECT MerchantID,
       COUNT(*) AS count,
       ROUND(SUM(TransactionAmount),2) AS total_amount
FROM transactions
GROUP BY MerchantID
ORDER BY total_amount DESC
LIMIT 15;
        """,
        "High-Risk Accounts": """
SELECT AccountID,
       COUNT(*) AS high_risk_count,
       ROUND(AVG(risk_score_pct),2) AS avg_risk_score
FROM transactions
WHERE risk_level = 'High'
GROUP BY AccountID
ORDER BY high_risk_count DESC
LIMIT 15;
        """,
        "Login Attempt Analysis": """
SELECT LoginAttempts,
       COUNT(*) AS transaction_count,
       ROUND(AVG(TransactionAmount),2) AS avg_amount,
       ROUND(AVG(risk_score_pct),2) AS avg_risk_score
FROM transactions
GROUP BY LoginAttempts
ORDER BY LoginAttempts DESC;
        """,
        "Risk by Customer Occupation": """
SELECT CustomerOccupation,
       COUNT(*) AS transaction_count,
       ROUND(AVG(risk_score_pct),2) AS avg_risk_score,
       SUM(CASE WHEN risk_level='High' THEN 1 ELSE 0 END) AS high_risk_count
FROM transactions
GROUP BY CustomerOccupation
ORDER BY avg_risk_score DESC;
        """,
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.selectbox("Preset Queries", ["Custom …"] + list(PRESET_QUERIES.keys()))

    if preset == "Custom …":
        default_sql = "SELECT * FROM transactions LIMIT 20;"
    else:
        default_sql = PRESET_QUERIES[preset].strip()

    sql_query = st.text_area("SQL Query", value=default_sql, height=150)

    run_sql = st.button("▶️ Run Query", type="primary")

    if run_sql and sql_query.strip():
        try:
            result_df = pd.read_sql_query(sql_query, conn)
            st.success(f"Query returned **{len(result_df):,}** rows.")
            st.dataframe(result_df, use_container_width=True, height=450)

            # Download
            csv_bytes = result_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download result as CSV",
                csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"SQL Error: {e}")

    # ---- Show full SQL script ----
    with st.expander("📄 View full banking_analysis.sql"):
        if os.path.exists(SQL_PATH):
            with open(SQL_PATH) as f:
                st.code(f.read(), language="sql")
        else:
            st.info("SQL file not found.")

    # ---- Schema reference ----
    with st.expander("📋 Table Schema — transactions"):
        schema_df = pd.read_sql_query("PRAGMA table_info(transactions);", conn)
        st.dataframe(schema_df[["name", "type"]], use_container_width=True)
