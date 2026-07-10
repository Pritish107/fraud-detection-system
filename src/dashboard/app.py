"""Streamlit dashboard: explore flagged transactions with SHAP explanations, and check
current drift/retraining status. Reads the same model artifacts and precomputed reports
the other modules produce — run those first (see README) before launching this.

Usage:
    .venv/Scripts/python.exe -m streamlit run src/dashboard/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explainability.explainer import FraudExplainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PATH = ROOT / "data" / "processed" / "example_transactions.json"
DRIFT_DIR = ROOT / "reports" / "drift"
FIG_DIR = ROOT / "reports" / "figures"

COLOR_FRAUD = "#e34948"
COLOR_NONFRAUD = "#2a78d6"
COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def load_explainer() -> FraudExplainer:
    return FraudExplainer()


@st.cache_data
def load_examples() -> list:
    if not EXAMPLES_PATH.exists():
        return []
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_drift_summary() -> dict:
    path = DRIFT_DIR / "drift_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_feature_drift() -> pd.DataFrame:
    path = DRIFT_DIR / "feature_drift.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def render_transaction_explorer() -> None:
    st.header("Transaction Explorer")
    st.caption("Pick a transaction and see the model's fraud probability, decision, "
               "and the top features driving that prediction.")

    fe = load_explainer()
    examples = load_examples()
    if not examples:
        st.warning("No example transactions found. Run `src/api/prepare_examples.py` first.")
        return

    labels = {
        f"{e['transaction_id']} — {e['category'].replace('_', ' ')} (actual: {e['actual_label']})": e
        for e in examples
    }
    choice = st.selectbox("Example transaction", list(labels.keys()))
    example = labels[choice]

    row = pd.DataFrame([example["features"]])
    result = fe.explain_row(row, top_n=8)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    col2.metric("Decision", result["decision"].replace("_", " ").title())
    col3.metric("Actual label", example["actual_label"].replace("_", " ").title())

    if result["decision"] == "fraud":
        st.error(f"Flagged as fraud (threshold {result['threshold']:.3f})")
    else:
        st.success(f"Not flagged (threshold {result['threshold']:.3f})")

    st.subheader("Top contributing features")
    feat_df = pd.DataFrame(result["top_features"])
    feat_df["direction"] = feat_df["shap_value"].apply(
        lambda v: "toward fraud" if v >= 0 else "away from fraud")
    feat_df["feature_value"] = feat_df["feature_value"].astype(str)
    feat_df = feat_df.sort_values("shap_value")

    st.bar_chart(
        feat_df.set_index("feature")["shap_value"],
        color=COLOR_FRAUD,
        horizontal=True,
    )
    st.dataframe(
        feat_df[["feature", "feature_value", "shap_value", "direction"]],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Full transaction payload"):
        st.json(example["features"])


def render_drift_monitoring() -> None:
    st.header("Drift Monitoring")
    st.caption("Feature-distribution drift (train vs. test) and out-of-sample "
               "performance over time — see reports/drift/drift_report.md for the "
               "full writeup.")

    summary = load_drift_summary()
    if not summary:
        st.warning("No drift summary found. Run `src/drift/generate_drift_report.py` first.")
        return

    if summary["retrain_recommended"]:
        st.error("**RETRAIN RECOMMENDED**")
        for reason in summary["trigger_reasons"]:
            st.markdown(f"- {reason}")
    else:
        st.success("**No retrain needed** — drift signals are within thresholds.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Features significantly drifted",
                f"{summary['n_significant']} / {summary['n_features']}",
                f"{summary['frac_significant']:.1%}")
    col2.metric("Out-of-sample PR-AUC change",
                f"{summary['last_window_pr_auc']:.3f}",
                f"{-summary['performance_relative_drop']:.1%} vs. first window")
    col3.metric("Prediction-distribution PSI", f"{summary['prediction_distribution_psi']:.3f}")

    st.subheader("Feature drift")
    fig_path = FIG_DIR / "drift_feature_psi.png"
    if fig_path.exists():
        st.image(str(fig_path))

    drift_table = load_feature_drift()
    if not drift_table.empty:
        with st.expander("Full feature drift table"):
            st.dataframe(drift_table, hide_index=True, use_container_width=True)

    st.subheader("Performance over time (out-of-sample)")
    fig_path = FIG_DIR / "drift_performance_over_time.png"
    if fig_path.exists():
        st.image(str(fig_path))

    st.subheader("Prediction distribution drift")
    fig_path = FIG_DIR / "drift_prediction_distribution.png"
    if fig_path.exists():
        st.image(str(fig_path))
    st.info(
        "Prediction-distribution PSI barely moved even though out-of-sample "
        "performance dropped substantially — this is concept drift (the "
        "relationship between features and fraud changed), which label-free "
        "monitoring alone would miss. See the full drift report for detail."
    )


def main() -> None:
    st.title("Fraud Detection — Dashboard")
    page = st.sidebar.radio("View", ["Transaction Explorer", "Drift Monitoring"])
    if page == "Transaction Explorer":
        render_transaction_explorer()
    else:
        render_drift_monitoring()


main()
