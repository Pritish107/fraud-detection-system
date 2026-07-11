"""Streamlit dashboard: explore flagged transactions with SHAP explanations, and check
current drift/retraining status. Reads the same model artifacts and precomputed reports
the other modules produce — run those first (see README) before launching this.

Usage:
    .venv/Scripts/python.exe -m streamlit run src/dashboard/app.py
"""

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explainability.explainer import FraudExplainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_EXAMPLES_PATH = ROOT / "data" / "processed" / "example_transactions.json"
REAL_EXAMPLES_PATH = ROOT / "data" / "processed" / "example_transactions_real_LOCAL_ONLY.json"
DRIFT_DIR = ROOT / "reports" / "drift"
FIG_DIR = ROOT / "reports" / "figures"

# dataviz skill categorical palette (references/palette.md) — same colors used
# throughout the project's static report figures, kept consistent here.
COLOR_FRAUD = "#e34948"
COLOR_NONFRAUD = "#2a78d6"
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#eda100"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#898781"

# Headline results from reports/models/{baseline_metrics,main_model_comparison,
# hyperparameter_tuning}.md and reports/eda/eda_report.md — the source of truth for
# these numbers is the pipeline that produced those reports; update here if the model
# is retrained. Main model is the Optuna-tuned LightGBM (hyperparameter_tuning.md).
FRAUD_RATE = 0.0350
BASELINE_PR_AUC = 0.189
MAIN_PR_AUC = 0.5619
MAIN_ROC_AUC = 0.9008

CATEGORY_ICONS = {
    "high_risk_synthetic": "\U0001F534",
    "low_risk_synthetic": "\U0001F7E2",
    "borderline_synthetic": "\U0001F7E1",
    "confident_fraud": "\U0001F534",
    "missed_fraud": "\U0001F7E0",
    "false_alarm": "\U0001F7E1",
    "typical_non_fraud": "\U0001F7E2",
}

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide",
                    initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown("""
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
        background: linear-gradient(135deg, rgba(42,120,214,0.10), rgba(227,73,72,0.10));
        border: 1px solid rgba(137,135,129,0.25);
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 1.8rem; }
    .hero p { margin: 0; opacity: 0.75; font-size: 0.98rem; }

    .metric-card {
        background: rgba(137,135,129,0.06);
        border: 1px solid rgba(137,135,129,0.25);
        border-radius: 12px; padding: 16px 18px; height: 100%;
    }
    .metric-card .label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
        opacity: 0.65; margin-bottom: 6px; font-weight: 600;
    }
    .metric-card .value { font-size: 1.65rem; font-weight: 700; line-height: 1.1; }
    .metric-card .delta { font-size: 0.82rem; margin-top: 6px; opacity: 0.85; }

    .badge {
        display: inline-block; padding: 7px 16px; border-radius: 999px;
        font-weight: 600; font-size: 0.95rem;
    }
    .badge-fraud {
        background: rgba(227,73,72,0.14); color: #e34948;
        border: 1px solid rgba(227,73,72,0.35);
    }
    .badge-safe {
        background: rgba(42,120,214,0.14); color: #2a78d6;
        border: 1px solid rgba(42,120,214,0.35);
    }
    .badge-critical {
        background: rgba(208,59,59,0.14); color: #d03b3b;
        border: 1px solid rgba(208,59,59,0.35);
    }
    .badge-good {
        background: rgba(12,163,12,0.14); color: #0ca30c;
        border: 1px solid rgba(12,163,12,0.35);
    }

    .sidebar-footer {
        font-size: 0.78rem; opacity: 0.6; padding: 10px 4px; line-height: 1.5;
    }

    /* Static report figures (matplotlib, light-surface #fcfcfb) are framed as an
       intentional "paper" card rather than left to clash raw against the dark app
       background. */
    .report-figure-card {
        background: #fcfcfb; border-radius: 12px; padding: 16px;
        border: 1px solid rgba(137,135,129,0.25);
    }
    .report-figure-card img { width: 100%; display: block; border-radius: 6px; }
    .report-figure-caption {
        font-size: 0.76rem; opacity: 0.55; margin-top: 6px; padding: 0 2px;
    }

    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)


def report_figure(path: Path, caption: str = "") -> None:
    """Embed a static report PNG (light surface, built for the standalone markdown
    reports) as a single self-contained HTML block, framed to read as an intentional
    inset rather than a dark-mode rendering bug."""
    if not path.exists():
        return
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    caption_html = f'<div class="report-figure-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="report-figure-card"><img src="data:image/png;base64,{b64}">'
        f'{caption_html}</div>',
        unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", delta_color: str = "") -> str:
    # Built as a single line deliberately: st.markdown runs its input through a
    # Markdown pass before allowing raw HTML through, and a multi-line f-string here
    # left an indented blank line before the closing </div> when delta was empty —
    # CommonMark reads 4-space indentation as a code block, so the closing tag
    # rendered as literal text instead of HTML. One line sidesteps that entirely.
    delta_html = f'<div class="delta" style="color:{delta_color}">{delta}</div>' if delta else ""
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{delta_html}</div>')


def badge(text: str, kind: str) -> None:
    icon = {"fraud": "\U0001F6A9", "safe": "✅", "critical": "⚠️", "good": "✅"}[kind]
    st.markdown(f'<span class="badge badge-{kind}">{icon} {text}</span>', unsafe_allow_html=True)


@st.cache_resource
def load_explainer() -> FraudExplainer:
    return FraudExplainer()


@st.cache_data
def load_examples(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


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


def make_gauge(proba: float, threshold: float) -> go.Figure:
    bar_color = COLOR_FRAUD if proba >= threshold else COLOR_NONFRAUD
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 42}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "rgba(137,135,129,0.08)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(42,120,214,0.12)"},
                {"range": [threshold * 100, 100], "color": "rgba(227,73,72,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLOR_CRITICAL, "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    # Plotly can't inherit the page's CSS theme (it renders to its own canvas/SVG), so
    # transparent backgrounds alone don't guarantee readable text — font/grid colors
    # are set explicitly here for the dark theme this dashboard is deployed with.
    fig.update_layout(
        height=240, margin=dict(l=25, r=25, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "color": "#e8e8e6"},
    )
    return fig


def make_shap_chart(feat_df: pd.DataFrame) -> go.Figure:
    colors = [COLOR_FRAUD if v >= 0 else COLOR_NONFRAUD for v in feat_df["shap_value"]]
    labels = [f"{f} = {v}" for f, v in zip(feat_df["feature"], feat_df["feature_value"])]
    fig = go.Figure(go.Bar(
        x=feat_df["shap_value"], y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in feat_df["shap_value"]], textposition="outside",
        textfont={"color": "#e8e8e6", "size": 11},
        hovertemplate="%{y}<br>SHAP: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(280, 34 * len(feat_df)),
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis_title="SHAP value (log-odds impact on prediction)",
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "size": 12, "color": "#e8e8e6"},
        xaxis=dict(gridcolor="rgba(232,232,230,0.18)", zerolinecolor="rgba(232,232,230,0.5)",
                   tickfont={"color": "#c3c2b7"}),
        yaxis=dict(autorange="reversed", tickfont={"color": "#e8e8e6"}),
    )
    return fig


def make_drift_chart(drift_table: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = drift_table.head(top_n).iloc[::-1]
    severity_color = {"significant": COLOR_FRAUD, "moderate": COLOR_WARNING, "none": COLOR_NONFRAUD}
    colors = [severity_color.get(s, COLOR_MUTED) for s in top["severity"]]
    fig = go.Figure(go.Bar(
        x=top["psi"], y=top["feature"], orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in top["psi"]], textposition="outside",
        textfont={"color": "#e8e8e6", "size": 11},
        hovertemplate="%{y}<br>PSI: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color="rgba(232,232,230,0.4)", line_width=1)
    fig.add_vline(x=0.2, line_dash="dash", line_color="rgba(232,232,230,0.4)", line_width=1)
    fig.update_layout(
        height=max(320, 28 * len(top)),
        margin=dict(l=10, r=30, t=10, b=40),
        xaxis_title="PSI (train vs. test) — dashed lines at 0.1 (moderate) and 0.2 (significant)",
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "size": 12, "color": "#e8e8e6"},
        xaxis=dict(gridcolor="rgba(232,232,230,0.18)", tickfont={"color": "#c3c2b7"}),
        yaxis=dict(tickfont={"color": "#e8e8e6"}),
    )
    return fig


def render_overview() -> None:
    st.markdown(
        '<div class="hero"><h1>Real-Time Fraud Detection</h1>'
        "<p>LightGBM classifier tuned for a 3.50% fraud prevalence, explained "
        "per-prediction with SHAP, and monitored for concept drift. Explore individual "
        "transactions or check the model's current drift status using the sidebar.</p>"
        "</div>",
        unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("Fraud prevalence", f"{FRAUD_RATE:.2%}", "of all transactions", COLOR_MUTED),
        ("Main model PR-AUC", f"{MAIN_PR_AUC:.3f}", f"vs. {BASELINE_PR_AUC:.3f} baseline", COLOR_GOOD),
        ("Main model ROC-AUC", f"{MAIN_ROC_AUC:.3f}", "LightGBM, test split", COLOR_MUTED),
        ("Improvement over baseline",
         f"{(MAIN_PR_AUC / BASELINE_PR_AUC - 1):.0%}",
         "relative PR-AUC gain", COLOR_GOOD),
    ]
    for col, (label, value, delta, color) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, delta, color), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Transaction Explorer")
        st.caption(
            "Score example transactions (fabricated, not real Kaggle data — see README) "
            "against the live model. See the fraud probability, decision, and the top "
            "SHAP-contributing features for each one."
        )
    with right:
        st.subheader("Drift Monitoring")
        st.caption(
            "Feature-distribution drift (PSI/KS) between the training window and the "
            "held-out test window, plus out-of-sample performance over time and a "
            "concrete retrain recommendation."
        )

    st.divider()
    summary = load_drift_summary()
    if summary:
        st.caption(
            f"Current drift status: "
            f"{'⚠️ retrain recommended' if summary['retrain_recommended'] else '✅ no retrain needed'} "
            f"— see the Drift Monitoring page for detail."
        )


def render_transaction_explorer() -> None:
    st.header("Transaction Explorer")
    st.caption("Pick a transaction and see the model's fraud probability, decision, "
               "and the top features driving that prediction.")

    fe = load_explainer()

    has_real = REAL_EXAMPLES_PATH.exists()
    if has_real:
        source = st.radio(
            "Data source", ["Fabricated demo data (shipped in repo)",
                             "Real IEEE-CIS transactions (local only)"],
            horizontal=True, label_visibility="collapsed",
        )
        is_real = source.startswith("Real")
    else:
        is_real = False
        st.caption(
            "Using fabricated demo data (see README). Run `src/api/prepare_examples.py` "
            "locally — with your own Kaggle access — to explore real held-out transactions "
            "instead; that option will appear here once generated."
        )

    examples = load_examples(REAL_EXAMPLES_PATH if is_real else SYNTHETIC_EXAMPLES_PATH)
    if not examples:
        st.warning("No example transactions found. Run "
                   "`src/api/generate_synthetic_examples.py` first.")
        return

    if is_real:
        badge("Real transactions — local only, never committed (see README)", "critical")
        st.write("")

    labels = {}
    for e in examples:
        icon = CATEGORY_ICONS.get(e["category"], "⚪")
        outcome = f" — actual: {e['actual_label']}" if is_real else ""
        label = f"{icon} {e['transaction_id']} — {e['category'].replace('_', ' ')}{outcome}"
        labels[label] = e
    choice = st.selectbox("Example transaction", list(labels.keys()))
    example = labels[choice]

    row = pd.DataFrame([example["features"]])
    result = fe.explain_row(row, top_n=8)

    gauge_col, info_col = st.columns([1, 1.4])
    with gauge_col:
        st.plotly_chart(make_gauge(result["fraud_probability"], result["threshold"]),
                         use_container_width=True, config={"displayModeBar": False})
    with info_col:
        st.write("")
        if result["decision"] == "fraud":
            badge("Flagged as fraud", "fraud")
        else:
            badge("Not flagged", "safe")
        st.write("")
        st.markdown(metric_card("Decision threshold", f"{result['threshold']:.3f}"),
                    unsafe_allow_html=True)
        st.write("")
        if is_real:
            match = (result["decision"] == "fraud") == (example["actual_label"] == "fraud")
            st.markdown(metric_card("Actual historical outcome",
                                     example["actual_label"].replace("_", " ").title(),
                                     "✓ model agrees" if match else "✗ model disagrees",
                                     COLOR_GOOD if match else COLOR_CRITICAL),
                        unsafe_allow_html=True)
        else:
            st.markdown(metric_card("Scenario (fabricated data)",
                                     example["category"].replace("_", " ").title()),
                        unsafe_allow_html=True)

    st.subheader("Top contributing features")
    st.caption("Positive SHAP values (red) push the prediction toward fraud; "
               "negative values (blue) push away from it.")
    feat_df = pd.DataFrame(result["top_features"])
    feat_df["feature_value"] = feat_df["feature_value"].astype(str)
    feat_df = feat_df.sort_values("shap_value")
    st.plotly_chart(make_shap_chart(feat_df), use_container_width=True,
                     config={"displayModeBar": False})

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
        badge("RETRAIN RECOMMENDED", "critical")
        for reason in summary["trigger_reasons"]:
            st.markdown(f"- {reason}")
    else:
        badge("No retrain needed — drift signals are within thresholds", "good")

    st.write("")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(metric_card(
            "Features significantly drifted",
            f"{summary['n_significant']} / {summary['n_features']}",
            f"{summary['frac_significant']:.1%}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card(
            "Out-of-sample PR-AUC",
            f"{summary['last_window_pr_auc']:.3f}",
            f"{-summary['performance_relative_drop']:.1%} vs. first window",
            COLOR_CRITICAL), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card(
            "Prediction-distribution PSI",
            f"{summary['prediction_distribution_psi']:.3f}",
            "label-free proxy signal"), unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["Feature Drift", "Performance Over Time", "Prediction Drift"])

    with tab1:
        drift_table = load_feature_drift()
        if not drift_table.empty:
            st.plotly_chart(make_drift_chart(drift_table), use_container_width=True,
                             config={"displayModeBar": False})
            with st.expander("Full feature drift table"):
                st.dataframe(drift_table, hide_index=True, use_container_width=True)

    with tab2:
        report_figure(FIG_DIR / "drift_performance_over_time.png",
                      "Out-of-sample PR-AUC in 3-day windows across validation+test — "
                      "never seen during training.")

    with tab3:
        report_figure(FIG_DIR / "drift_prediction_distribution.png",
                      "Predicted fraud-probability distribution, train vs. test.")
        st.info(
            "Prediction-distribution PSI barely moved even though out-of-sample "
            "performance dropped substantially — this is concept drift (the "
            "relationship between features and fraud changed), which label-free "
            "monitoring alone would miss. See the full drift report for detail."
        )


def main() -> None:
    inject_css()
    st.sidebar.title("\U0001F6E1️ Fraud Detection")
    page = st.sidebar.radio("View", ["Overview", "Transaction Explorer", "Drift Monitoring"],
                             label_visibility="collapsed")

    if page == "Overview":
        render_overview()
    elif page == "Transaction Explorer":
        render_transaction_explorer()
    else:
        render_drift_monitoring()

    try:
        fe = load_explainer()
        st.sidebar.markdown(
            '<div class="sidebar-footer">'
            "<b>Model:</b> LightGBM<br>"
            f"<b>Threshold:</b> {fe.threshold:.3f}<br>"
            f"<b>Best iteration:</b> {fe.meta['best_iteration']}<br>"
            f"<b>Features:</b> {len(fe.feature_cols)}"
            "</div>",
            unsafe_allow_html=True)
    except Exception:
        pass


main()
