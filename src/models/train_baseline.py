"""Train the baseline Logistic Regression fraud classifier.

Uses class_weight="balanced" since an unweighted LR at 3.5% fraud prevalence would
trivially predict "not fraud" almost everywhere. Evaluated on PR-AUC (the primary metric
for this project, per the problem statement) rather than accuracy, plus ROC-AUC and a
classification report at the default 0.5 threshold for reference.

Usage:
    .venv/Scripts/python.exe src/models/train_baseline.py
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import load_merged  # noqa: E402
from models.preprocessing import build_preprocessor  # noqa: E402
from models.split import time_based_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "models"
FIG_DIR = ROOT / "reports" / "figures"

COLOR_FRAUD = "#e34948"
COLOR_GRID = "#e1e0d9"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged train data...")
    df = load_merged("train")

    print("Time-based split...")
    train, val, test = time_based_split(df)
    print(f"  train: {len(train):,} ({train['isFraud'].mean():.2%} fraud)")
    print(f"  val:   {len(val):,} ({val['isFraud'].mean():.2%} fraud)")
    print(f"  test:  {len(test):,} ({test['isFraud'].mean():.2%} fraud)")

    preprocessor, numeric, categorical = build_preprocessor(train)
    print(f"  {len(numeric)} numeric + {len(categorical)} categorical features")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)),
    ])

    feature_cols = numeric + categorical
    X_train, y_train = train[feature_cols], train["isFraud"]
    X_val, y_val = val[feature_cols], val["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]

    print("Training Logistic Regression baseline...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  fit in {time.time() - start:.1f}s")

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    val_pr_auc = average_precision_score(y_val, val_proba)
    val_roc_auc = roc_auc_score(y_val, val_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)
    test_roc_auc = roc_auc_score(y_test, test_proba)

    val_report = classification_report(y_val, (val_proba >= 0.5).astype(int),
                                        target_names=["Non-Fraud", "Fraud"])
    print(f"\nValidation PR-AUC: {val_pr_auc:.4f}  ROC-AUC: {val_roc_auc:.4f}")
    print(f"Test PR-AUC:       {test_pr_auc:.4f}  ROC-AUC: {test_roc_auc:.4f}")
    print(f"\nValidation classification report (threshold=0.5):\n{val_report}")

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_val, val_proba, name="Baseline LR (val)",
                                             color=COLOR_FRAUD, ax=ax)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.set_title("Baseline Logistic Regression: Precision-Recall (validation)",
                  loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_pr_curve.png", dpi=150)
    plt.close(fig)

    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    report_path = REPORT_DIR / "baseline_metrics.md"
    report_path.write_text(
        "# Baseline Model — Logistic Regression\n\n"
        "Time-based split (train on earliest 70%, validate on next 15%, test on most "
        "recent 15%). `class_weight=\"balanced\"` used since fraud is 3.50% of "
        "transactions.\n\n"
        "## Metrics\n\n"
        "| Split | PR-AUC | ROC-AUC |\n"
        "|---|---|---|\n"
        f"| Validation | {val_pr_auc:.4f} | {val_roc_auc:.4f} |\n"
        f"| Test | {test_pr_auc:.4f} | {test_roc_auc:.4f} |\n\n"
        "## Validation classification report (threshold = 0.5)\n\n"
        f"```\n{val_report}\n```\n\n"
        "![Precision-Recall curve](../figures/baseline_pr_curve.png)\n\n"
        "PR-AUC (not accuracy) is the headline metric: at 3.5% fraud prevalence, a "
        "model that never flags fraud still scores ~96.5% accuracy. This baseline "
        "establishes the floor that the main LightGBM/XGBoost model "
        "(with a proper imbalance-handling comparison) needs to beat.\n",
        encoding="utf-8",
    )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
