"""Generate global + local SHAP explainability report for the main LightGBM model.

Global: which features drive fraud predictions overall (summary beeswarm + mean |SHAP|
importance), computed on a sample of the test set for tractability.
Local: waterfall explanations for three representative transactions — a confidently
caught fraud, a missed fraud (false negative), and a false alarm (false positive) — so
the report demonstrates the "human-readable justification per flagged transaction"
deliverable, not just aggregate importance.

Usage:
    .venv/Scripts/python.exe src/explainability/generate_shap_report.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import load_merged  # noqa: E402
from explainability.explainer import FraudExplainer  # noqa: E402
from models.split import time_based_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports" / "explainability"

SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def plot_global(shap_values, X, fe: FraudExplainer) -> None:
    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_importance.png", dpi=150)
    plt.close()


def plot_local(fe: FraudExplainer, row, fig_name: str) -> dict:
    result = fe.explain_row(row, top_n=8)
    X = fe._prepare(row)
    sv = fe._explainer.shap_values(X)[0]
    base = fe._explainer.expected_value

    expl = shap.Explanation(values=sv, base_values=base, data=X.iloc[0].values,
                             feature_names=fe.feature_cols)
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close()
    return result


def format_local_section(title: str, description: str, txn_id, actual_label: str,
                          result: dict, fig_name: str) -> list:
    lines = [f"## {title}", "", description, "", f"- TransactionID: {txn_id}",
              f"- Actual label: {actual_label}",
              f"- Predicted fraud probability: {result['fraud_probability']:.3f}",
              f"- Decision at threshold {result['threshold']:.3f}: **{result['decision']}**",
              "", "Top contributing features (SHAP value in log-odds; positive pushes "
              "toward fraud, negative pushes away):", ""]
    for f in result["top_features"]:
        sign = "+" if f["shap_value"] >= 0 else ""
        lines.append(f"- `{f['feature']}` = {f['feature_value']} → {sign}{f['shap_value']:.3f}")
    lines += ["", f"![{title}](../figures/{fig_name})", ""]
    return lines


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and splitting (time-based, same as training)...")
    df = load_merged("train")
    _, _, test = time_based_split(df)

    fe = FraudExplainer()
    test_proba = fe.predict_proba(test)
    test = test.assign(_proba=test_proba)

    print("Sampling test set for global SHAP...")
    sample = test.sample(n=min(SAMPLE_SIZE, len(test)), random_state=RANDOM_STATE)
    shap_values, X_sample = fe.shap_values(sample)

    print("Global SHAP plots...")
    plot_global(shap_values, X_sample, fe)

    print("Local explanations...")
    fraud_mask = test["isFraud"] == 1
    nonfraud_mask = test["isFraud"] == 0

    true_positive = test[fraud_mask & (test["_proba"] >= fe.threshold)].sort_values(
        "_proba", ascending=False).head(1)
    false_negative = test[fraud_mask & (test["_proba"] < fe.threshold)].sort_values(
        "_proba", ascending=True).head(1)
    false_positive = test[nonfraud_mask & (test["_proba"] >= fe.threshold)].sort_values(
        "_proba", ascending=False).head(1)

    tp_result = plot_local(fe, true_positive, "shap_local_true_positive.png")
    fn_result = plot_local(fe, false_negative, "shap_local_false_negative.png")
    fp_result = plot_local(fe, false_positive, "shap_local_false_positive.png")

    print("Writing report...")
    lines = [
        "# Explainability Report — SHAP",
        "",
        "SHAP (TreeExplainer) values for the main LightGBM model, computed on the "
        "held-out test split. Values are in log-odds space (the model's raw output "
        "before the sigmoid) — positive SHAP values push a prediction toward fraud, "
        "negative values push away from it.",
        "",
        "## Global feature importance",
        "",
        f"Computed on a random sample of {len(X_sample):,} test transactions.",
        "",
        "![SHAP summary](../figures/shap_summary.png)",
        "",
        "![SHAP importance](../figures/shap_importance.png)",
        "",
        "The dominant drivers are the anonymized `V*`/`C*`/`D*` engineered features "
        "(consistent with the correlation analysis in the EDA report), plus "
        "`TransactionAmt`, `card1`/`card2` (card identifiers), and `addr1`. Kaggle "
        "doesn't disclose what the `V*` columns represent, which is a real limitation "
        "for regulatory explanations in production — see Limitations in the main "
        "README.",
        "",
        "## Local explanations",
        "",
        "Three representative transactions, to show the explanation the API returns "
        "for an individual flagged (or missed) transaction — not just aggregate "
        "importance.",
        "",
    ]
    lines += format_local_section(
        "Correctly caught fraud (true positive)",
        "Highest-confidence correct fraud catch in the test set.",
        true_positive["TransactionID"].iloc[0], "Fraud", tp_result,
        "shap_local_true_positive.png")
    lines += format_local_section(
        "Missed fraud (false negative)",
        "Actual fraud the model scored lowest — illustrates the recall gap the "
        "precision/recall tradeoff in the main model report accepts.",
        false_negative["TransactionID"].iloc[0], "Fraud", fn_result,
        "shap_local_false_negative.png")
    lines += format_local_section(
        "False alarm (false positive)",
        "Legitimate transaction the model flagged most confidently — the cost side "
        "of lowering the decision threshold for recall.",
        false_positive["TransactionID"].iloc[0], "Not Fraud", fp_result,
        "shap_local_false_positive.png")

    (REPORT_DIR / "shap_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {REPORT_DIR / 'shap_report.md'}")


if __name__ == "__main__":
    main()
