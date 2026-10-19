"""
visualization.py
----------------
Reusable Plotly chart-builder functions used by the Streamlit app.
Each function accepts a DataFrame and returns a Plotly figure so the
app can call fig.show() or st.plotly_chart(fig).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette — consistent across all charts
# ---------------------------------------------------------------------------
COLORS = {
    "primary":  "#3b82f6",   # blue
    "success":  "#22c55e",   # green
    "warning":  "#f59e0b",   # amber
    "danger":   "#ef4444",   # red
    "muted":    "#94a3b8",   # slate
    "bg":       "#0f172a",   # dark background (unused in light theme)
    "Low":      "#22c55e",
    "Medium":   "#f59e0b",
    "High":     "#ef4444",
}

RISK_COLOR_MAP = {"Low": COLORS["Low"], "Medium": COLORS["Medium"], "High": COLORS["High"]}

TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Dashboard charts
# ---------------------------------------------------------------------------

def transaction_amount_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of transaction amounts."""
    fig = px.histogram(
        df,
        x="TransactionAmount",
        nbins=50,
        title="Transaction Amount Distribution",
        labels={"TransactionAmount": "Amount ($)"},
        color_discrete_sequence=[COLORS["primary"]],
        template=TEMPLATE,
    )
    fig.update_layout(bargap=0.05)
    return fig


def transaction_trend_line(df: pd.DataFrame) -> go.Figure:
    """Daily transaction count over time."""
    daily = (
        df.set_index("TransactionDate")
        .resample("D")["TransactionAmount"]
        .count()
        .reset_index()
        .rename(columns={"TransactionAmount": "count"})
    )
    fig = px.line(
        daily,
        x="TransactionDate",
        y="count",
        title="Daily Transaction Volume",
        labels={"TransactionDate": "Date", "count": "Transactions"},
        color_discrete_sequence=[COLORS["primary"]],
        template=TEMPLATE,
    )
    return fig


def transaction_amount_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly total transaction amount."""
    monthly = (
        df.groupby(df["TransactionDate"].dt.to_period("M"))["TransactionAmount"]
        .sum()
        .reset_index()
    )
    monthly["TransactionDate"] = monthly["TransactionDate"].astype(str)
    fig = px.bar(
        monthly,
        x="TransactionDate",
        y="TransactionAmount",
        title="Monthly Total Transaction Value ($)",
        labels={"TransactionDate": "Month", "TransactionAmount": "Total Amount ($)"},
        color_discrete_sequence=[COLORS["primary"]],
        template=TEMPLATE,
    )
    return fig


def transaction_type_pie(df: pd.DataFrame) -> go.Figure:
    """Credit vs Debit breakdown."""
    counts = df["TransactionType"].value_counts().reset_index()
    counts.columns = ["TransactionType", "count"]
    fig = px.pie(
        counts,
        names="TransactionType",
        values="count",
        title="Transaction Type Distribution",
        color_discrete_sequence=[COLORS["primary"], COLORS["success"]],
        template=TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def channel_bar(df: pd.DataFrame) -> go.Figure:
    """Transaction count by Channel."""
    counts = df["Channel"].value_counts().reset_index()
    counts.columns = ["Channel", "count"]
    fig = px.bar(
        counts,
        x="Channel",
        y="count",
        title="Transactions by Channel",
        labels={"count": "Number of Transactions"},
        color="Channel",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=TEMPLATE,
    )
    return fig


def location_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Top locations by transaction count."""
    counts = (
        df["Location"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["Location", "count"]
    fig = px.bar(
        counts,
        x="count",
        y="Location",
        orientation="h",
        title=f"Top {top_n} Locations by Transaction Count",
        labels={"count": "Transactions", "Location": ""},
        color_discrete_sequence=[COLORS["primary"]],
        template=TEMPLATE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def occupation_bar(df: pd.DataFrame) -> go.Figure:
    """Transaction count by customer occupation."""
    counts = df["CustomerOccupation"].value_counts().reset_index()
    counts.columns = ["Occupation", "count"]
    fig = px.bar(
        counts,
        x="Occupation",
        y="count",
        title="Transactions by Customer Occupation",
        labels={"count": "Transactions", "Occupation": "Occupation"},
        color="Occupation",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template=TEMPLATE,
    )
    return fig


def age_group_bar(df: pd.DataFrame) -> go.Figure:
    """Transaction count by age group."""
    if "age_group" not in df.columns:
        bins   = [0, 25, 35, 50, 65, 120]
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        df = df.copy()
        df["age_group"] = pd.cut(df["CustomerAge"], bins=bins, labels=labels).astype(str)

    counts = df["age_group"].value_counts().reset_index()
    counts.columns = ["Age Group", "count"]
    order  = ["18-25", "26-35", "36-50", "51-65", "65+"]
    counts["Age Group"] = pd.Categorical(counts["Age Group"], categories=order, ordered=True)
    counts = counts.sort_values("Age Group")

    fig = px.bar(
        counts,
        x="Age Group",
        y="count",
        title="Transactions by Customer Age Group",
        labels={"count": "Transactions"},
        color_discrete_sequence=[COLORS["primary"]],
        template=TEMPLATE,
    )
    return fig


def risk_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Risk level breakdown."""
    if "risk_level" not in df.columns:
        return go.Figure()
    counts = df["risk_level"].value_counts().reset_index()
    counts.columns = ["Risk Level", "count"]
    fig = px.pie(
        counts,
        names="Risk Level",
        values="count",
        title="Risk Level Distribution",
        color="Risk Level",
        color_discrete_map=RISK_COLOR_MAP,
        template=TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def risk_score_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of risk_score_pct across all transactions."""
    if "risk_score_pct" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df,
        x="risk_score_pct",
        nbins=40,
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        title="Risk Score Distribution",
        labels={"risk_score_pct": "Risk Score (%)", "risk_level": "Risk Level"},
        template=TEMPLATE,
    )
    return fig


def login_attempts_bar(df: pd.DataFrame) -> go.Figure:
    """Distribution of login attempts."""
    counts = df["LoginAttempts"].value_counts().sort_index().reset_index()
    counts.columns = ["Login Attempts", "count"]
    fig = px.bar(
        counts,
        x="Login Attempts",
        y="count",
        title="Distribution of Login Attempts",
        labels={"count": "Transactions"},
        color_discrete_sequence=[COLORS["warning"]],
        template=TEMPLATE,
    )
    return fig


def transaction_duration_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of transaction duration by channel."""
    fig = px.box(
        df,
        x="Channel",
        y="TransactionDuration",
        title="Transaction Duration by Channel",
        labels={"TransactionDuration": "Duration (s)"},
        color="Channel",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=TEMPLATE,
    )
    return fig


def balance_vs_amount_scatter(df: pd.DataFrame) -> go.Figure:
    """Account balance vs transaction amount scatter coloured by risk."""
    if "risk_level" not in df.columns:
        color_col = None
        color_map = None
    else:
        color_col = "risk_level"
        color_map = RISK_COLOR_MAP

    sample = df.sample(min(800, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="AccountBalance",
        y="TransactionAmount",
        color=color_col,
        color_discrete_map=color_map,
        title="Account Balance vs Transaction Amount",
        labels={
            "AccountBalance":    "Account Balance ($)",
            "TransactionAmount": "Transaction Amount ($)",
        },
        opacity=0.6,
        template=TEMPLATE,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix of key numerical features."""
    num_cols = [
        "TransactionAmount", "AccountBalance", "TransactionDuration",
        "LoginAttempts", "CustomerAge",
    ]
    if "risk_score_pct" in df.columns:
        num_cols.append("risk_score_pct")

    corr = df[num_cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Correlation Matrix (Key Numeric Features)",
        template=TEMPLATE,
    )
    return fig


def feature_importance_bar(fi_dict: dict, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of global feature importances."""
    items = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*items)

    # Human-readable labels
    try:
        from prediction import FEATURE_LABELS
        features = [FEATURE_LABELS.get(f, f) for f in features]
    except ImportError:
        pass

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(features),
        orientation="h",
        marker_color=COLORS["primary"],
    ))
    fig.update_layout(
        title=f"Top {top_n} Risk-Contributing Features (Global)",
        xaxis_title="Permutation Importance",
        yaxis={"categoryorder": "total ascending"},
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Account-level charts
# ---------------------------------------------------------------------------

def account_amount_over_time(df: pd.DataFrame, account_id: str) -> go.Figure:
    """Transaction amounts for a single account over time."""
    acct_df = df[df["AccountID"] == account_id].sort_values("TransactionDate")
    fig = px.scatter(
        acct_df,
        x="TransactionDate",
        y="TransactionAmount",
        color="risk_level" if "risk_level" in acct_df.columns else None,
        color_discrete_map=RISK_COLOR_MAP,
        title=f"Transaction History — {account_id}",
        labels={"TransactionAmount": "Amount ($)", "TransactionDate": "Date"},
        template=TEMPLATE,
    )
    return fig


def account_risk_gauge(risk_pct: float) -> go.Figure:
    """Gauge chart showing a single transaction's risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        title={"text": "Risk Score (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": COLORS["danger"] if risk_pct >= 70
                              else COLORS["warning"] if risk_pct >= 40
                              else COLORS["success"]},
            "steps": [
                {"range": [0,  40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(template=TEMPLATE)
    return fig
