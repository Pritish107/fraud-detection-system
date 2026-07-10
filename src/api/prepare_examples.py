"""Precompute a handful of REAL example transactions (from the actual held-out test
split) for local exploration of the API/dashboard against real fraud cases.

*** LOCAL USE ONLY — DO NOT COMMIT THIS SCRIPT'S OUTPUT. ***
IEEE-CIS's Kaggle competition rules restrict redistributing the dataset outside the
competition, so its output is written to a path .gitignore excludes by pattern (not
example_transactions.json — see generate_synthetic_examples.py for the fabricated data
that actually ships in the repo and backs the API/dashboard by default).

Usage:
    .venv/Scripts/python.exe src/api/prepare_examples.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import load_merged  # noqa: E402
from explainability.explainer import FraudExplainer  # noqa: E402
from models.split import time_based_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "example_transactions_real_LOCAL_ONLY.json"


def row_to_json_safe(row) -> dict:
    out = {}
    for k, v in row.items():
        if pd_isna(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def pd_isna(v) -> bool:
    try:
        return bool(v != v)  # NaN != NaN
    except Exception:
        return v is None


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_merged("train")
    _, _, test = time_based_split(df)

    fe = FraudExplainer()
    test = test.assign(_proba=fe.predict_proba(test))

    fraud = test[test["isFraud"] == 1]
    nonfraud = test[test["isFraud"] == 0]

    picks = {
        "confident_fraud": fraud.sort_values("_proba", ascending=False).head(2),
        "missed_fraud": fraud.sort_values("_proba", ascending=True).head(2),
        "false_alarm": nonfraud.sort_values("_proba", ascending=False).head(2),
        "typical_non_fraud": nonfraud.sample(n=2, random_state=42),
    }

    examples = []
    for category, rows in picks.items():
        for _, row in rows.iterrows():
            examples.append({
                "transaction_id": int(row["TransactionID"]),
                "category": category,
                "actual_label": "fraud" if row["isFraud"] == 1 else "not_fraud",
                "model_proba_at_export": float(row["_proba"]),
                "features": row_to_json_safe(row[fe.feature_cols]),
            })

    OUT_PATH.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Saved {len(examples)} example transactions to {OUT_PATH}")


if __name__ == "__main__":
    main()
