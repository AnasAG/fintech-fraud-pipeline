"""
Streamlit monitoring dashboard.

What this shows:
  A live fraud ops view using the test set replayed as a transaction stream.
  Demonstrates the kind of monitoring a deployed fraud model needs.

Panels:
  1. Score Distribution  — histogram of fraud probabilities (skewed = confident)
  2. Decision Breakdown  — APPROVE / REVIEW / DECLINE over time
  3. Drift Detector      — toggle concept drift and watch score distribution shift
  4. Alert Zone          — fraud rate > 2σ above baseline triggers a warning
  5. Model Metadata      — version, training metrics, champion model info

Run:
  streamlit run src/monitoring/dashboard.py
  or: make dashboard
"""

import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.monitoring.simulate_stream import (
    compute_window_metrics,
    inject_concept_drift,
    load_test_set,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Live Monitor",
    page_icon="🔍",
    layout="wide",
)


# ── Load data and model ────────────────────────────────────────────────────────
@st.cache_data
def get_test_data(drift: bool = False) -> pd.DataFrame:
    df = load_test_set()
    if drift:
        df = inject_concept_drift(df, drift_fraction=0.15)
    return df


@st.cache_resource
def load_predictor():
    from src.api.predictor import FraudPredictor
    p = FraudPredictor()
    try:
        p.load()
        return p
    except FileNotFoundError:
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Fraud Detection — Live Monitor")
st.caption("Simulated transaction stream from IEEE-CIS test set · Replayed at configurable speed")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    inject_drift = st.toggle("Inject concept drift", value=False)
    batch_size = st.slider("Batch size (txns per tick)", 50, 500, 200, step=50)
    replay_speed = st.select_slider(
        "Replay speed",
        options=["slow (2s)", "normal (1s)", "fast (0.3s)"],
        value="normal (1s)",
    )
    speed_map = {"slow (2s)": 2.0, "normal (1s)": 1.0, "fast (0.3s)": 0.3}
    tick_delay = speed_map[replay_speed]

    st.markdown("---")
    st.subheader("Decision Thresholds")
    st.caption("These are business parameters, not model parameters")
    threshold_approve = st.slider("Approve below", 0.1, 0.5, 0.3, 0.05)
    threshold_decline = st.slider("Decline above", 0.5, 0.9, 0.7, 0.05)

# ── Load data ─────────────────────────────────────────────────────────────────
predictor = load_predictor()
test_df = get_test_data(drift=inject_drift)

if predictor is None:
    st.error(
        "Model not loaded. Run `make train` to generate the champion model, then restart the dashboard."
    )
    st.stop()

# ── Session state for accumulated predictions ─────────────────────────────────
if "predictions_log" not in st.session_state or inject_drift != st.session_state.get("last_drift"):
    st.session_state.predictions_log = pd.DataFrame()
    st.session_state.batch_idx = 0
    st.session_state.last_drift = inject_drift

# ── Metric row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

metrics_placeholder = {
    "total": col1.empty(),
    "approve_pct": col2.empty(),
    "review_pct": col3.empty(),
    "decline_pct": col4.empty(),
    "fraud_rate": col5.empty(),
}

# ── Chart placeholders ────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Score Distribution")
    score_chart = st.empty()

with col_right:
    st.subheader("Decision Breakdown Over Time")
    decision_chart = st.empty()

alert_placeholder = st.empty()

st.markdown("---")
col_drift, col_meta = st.columns(2)
with col_drift:
    st.subheader("Drift Indicator")
    drift_chart = st.empty()

with col_meta:
    st.subheader("Model Metadata")
    st.json({
        "model_type": predictor.model_type,
        "model_version": predictor.model_version,
        "val_pr_auc": predictor.manifest.get("val_pr_auc"),
        "test_pr_auc": predictor.manifest.get("test_pr_auc"),
        "n_features": len(predictor.feature_columns),
        "thresholds": {"approve": threshold_approve, "decline": threshold_decline},
        "drift_injected": inject_drift,
    })

# ── Simulation loop ───────────────────────────────────────────────────────────
from src.features.build_features import build_features
import numpy as np

start_btn = st.button("▶ Start simulation", type="primary")

if start_btn:
    while True:
        start = st.session_state.batch_idx * batch_size
        batch = test_df.iloc[start : start + batch_size]

        if batch.empty:
            alert_placeholder.success("Simulation complete — all test transactions replayed.")
            break

        # Score entire batch in one model call (not row-by-row)
        try:
            X_batch, _, _ = build_features(batch.copy(), fit=False, encoders=predictor.encoders)
            for col in predictor.feature_columns:
                if col not in X_batch.columns:
                    X_batch[col] = np.nan
            X_batch = X_batch[predictor.feature_columns].fillna(-999)
            probs = predictor.model.predict_proba(X_batch)[:, 1]

            batch_results = []
            for prob, (_, row) in zip(probs, batch.iterrows()):
                if prob < threshold_approve:
                    decision = "APPROVE"
                elif prob > threshold_decline:
                    decision = "DECLINE"
                else:
                    decision = "REVIEW"
                batch_results.append({
                    "fraud_probability": round(float(prob), 4),
                    "decision": decision,
                    "isFraud": row.get("isFraud"),
                    "TransactionAmt": row.get("TransactionAmt"),
                    "batch": st.session_state.batch_idx,
                })

            new_rows = pd.DataFrame(batch_results)
            st.session_state.predictions_log = pd.concat(
                [st.session_state.predictions_log, new_rows], ignore_index=True
            )
        except Exception as e:
            alert_placeholder.warning(f"Batch scoring error: {e}")
            break

        st.session_state.batch_idx += 1
        log = st.session_state.predictions_log
        window_metrics = compute_window_metrics(log)

        # ── Update metrics in-place (no page flash) ───────────────────────────
        metrics_placeholder["total"].metric("Total scored", f"{len(log):,}")
        metrics_placeholder["approve_pct"].metric("APPROVE", f"{window_metrics.get('pct_approve', 0):.1%}")
        metrics_placeholder["review_pct"].metric("REVIEW", f"{window_metrics.get('pct_review', 0):.1%}")
        metrics_placeholder["decline_pct"].metric("DECLINE", f"{window_metrics.get('pct_decline', 0):.1%}")
        fraud_rate = window_metrics.get("actual_fraud_rate")
        metrics_placeholder["fraud_rate"].metric(
            "Actual fraud rate",
            f"{fraud_rate:.2%}" if fraud_rate is not None else "—"
        )

        # ── Score distribution ────────────────────────────────────────────────
        fig_hist = px.histogram(
            log, x="fraud_probability", nbins=50,
            color_discrete_sequence=["#EF553B"],
            labels={"fraud_probability": "Fraud Probability"},
        )
        fig_hist.add_vline(x=threshold_approve, line_dash="dash", line_color="green", annotation_text="Approve")
        fig_hist.add_vline(x=threshold_decline, line_dash="dash", line_color="red", annotation_text="Decline")
        score_chart.plotly_chart(fig_hist, use_container_width=True)

        # ── Decision breakdown ────────────────────────────────────────────────
        decision_by_batch = log.groupby(["batch", "decision"]).size().reset_index(name="count")
        fig_decisions = px.bar(
            decision_by_batch, x="batch", y="count", color="decision",
            color_discrete_map={"APPROVE": "#00CC96", "REVIEW": "#FFA15A", "DECLINE": "#EF553B"},
            barmode="stack",
        )
        decision_chart.plotly_chart(fig_decisions, use_container_width=True)

        # ── Drift indicator ───────────────────────────────────────────────────
        rolling_mean = log.groupby("batch")["fraud_probability"].mean().reset_index()
        baseline = rolling_mean["fraud_probability"].iloc[:3].mean() if len(rolling_mean) >= 3 else None
        fig_drift = px.line(
            rolling_mean, x="batch", y="fraud_probability",
            labels={"fraud_probability": "Mean fraud probability", "batch": "Batch"},
        )
        if baseline:
            fig_drift.add_hline(
                y=baseline + 2 * rolling_mean["fraud_probability"].std(),
                line_dash="dash", line_color="red", annotation_text="2σ alert threshold",
            )
        drift_chart.plotly_chart(fig_drift, use_container_width=True)

        # ── Alert zone ────────────────────────────────────────────────────────
        if baseline and len(rolling_mean) > 3:
            current = rolling_mean["fraud_probability"].iloc[-1]
            threshold_2sigma = baseline + 2 * rolling_mean["fraud_probability"].std()
            if current > threshold_2sigma:
                alert_placeholder.error(
                    f"ALERT: Mean fraud score ({current:.3f}) exceeds 2σ baseline "
                    f"({threshold_2sigma:.3f}). Possible fraud spike or model drift."
                )
            else:
                alert_placeholder.success("Score distribution within normal range.")

        time.sleep(tick_delay)
