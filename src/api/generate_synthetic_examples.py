"""Generate FABRICATED example transactions for the repo's committed
data/processed/example_transactions.json — safe to publish, unlike
prepare_examples.py's output (which pulls real rows from the Kaggle dataset and must
stay local/gitignored, since IEEE-CIS's competition rules restrict redistributing the
data outside the competition).

These values are randomly sampled from plausible ranges per column, NOT copied from
any real transaction. They exist purely so a fresh clone (and CI) can exercise
/examples, /predict, and the dashboard without needing Kaggle access. Since there's no
real ground truth for a fabricated transaction, "category" reflects the scenario each
example was constructed to illustrate (and is confirmed against the real model's own
prediction) rather than a verified historical outcome.

Usage:
    .venv/Scripts/python.exe src/api/generate_synthetic_examples.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from explainability.explainer import FraudExplainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "example_transactions.json"

RANDOM_STATE = 42

CATEGORICAL_CHOICES = {
    "ProductCD": ["W", "C", "H", "R", "S"],
    "card4": ["visa", "mastercard", "american express", "discover"],
    "card6": ["debit", "credit"],
    "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", None],
    "R_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", None],
    "id_30": ["Windows 10", "Mac OS X 10_15", "Android", "iOS", None],
    "id_31": ["chrome generic", "safari generic", "firefox generic", "edge generic", None],
    "id_33": ["1920x1080", "1366x768", "390x844", None],
    "id_34": ["match_status:2", "match_status:1", "match_status:0", None],
    "DeviceType": ["mobile", "desktop"],
    "DeviceInfo": ["Generic Windows PC", "Generic Android Device", "Generic iOS Device",
                   "Generic Mac", None],
}
BINARY_TF_COLS_PREFIX = "M"
BINARY_FOUND_COLS = ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
                      "id_35", "id_36", "id_37", "id_38"]


def sample_numeric(col: str, rng: random.Random):
    if rng.random() < 0.3:
        return None  # mimic realistic missingness (see EDA report)
    if col == "TransactionAmt":
        return round(rng.uniform(5, 1500), 2)
    if col in ("card1", "addr1"):
        return rng.randint(1000, 18000)
    if col in ("card2", "card5", "addr2"):
        return rng.randint(100, 600)
    if col == "card3":
        return rng.randint(100, 231)
    if col in ("dist1", "dist2"):
        return rng.randint(0, 100)
    if col.startswith("C"):
        return rng.randint(0, 100)
    if col.startswith("D"):
        return round(rng.uniform(0, 300), 1)
    if col.startswith("V"):
        return round(rng.uniform(0, 20), 1)
    if col.startswith("id_"):
        return round(rng.uniform(0, 500), 1)
    return round(rng.uniform(0, 100), 1)


def sample_categorical(col: str, rng: random.Random):
    if col in CATEGORICAL_CHOICES:
        return rng.choice(CATEGORICAL_CHOICES[col])
    if col.startswith(BINARY_TF_COLS_PREFIX):
        return rng.choice(["T", "F", None])
    if col in BINARY_FOUND_COLS:
        return rng.choice(["Found", "NotFound", None])
    return rng.choice(["F", None])


def generate_row(fe: FraudExplainer, rng: random.Random) -> dict:
    row = {}
    for col in fe.numeric_features:
        row[col] = sample_numeric(col, rng)
    for col in fe.categorical_features:
        row[col] = sample_categorical(col, rng)
    return row


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fe = FraudExplainer()
    rng = random.Random(RANDOM_STATE)

    print("Sampling candidate synthetic transactions...")
    candidates = [generate_row(fe, rng) for _ in range(3000)]
    df = pd.DataFrame(candidates)
    proba = fe.predict_proba(df)

    order = proba.sort_values()
    low = order.index[:2]
    high = order.index[-2:]
    mid = (proba - fe.threshold).abs().sort_values().index[:2]

    picks = {
        "high_risk_synthetic": (high, "fraud"),
        "low_risk_synthetic": (low, "not_fraud"),
        "borderline_synthetic": (mid, "unlabeled (synthetic, near threshold)"),
    }

    examples = []
    for category, (idxs, label) in picks.items():
        for i in idxs:
            row = df.loc[i].to_dict()
            row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            examples.append({
                "transaction_id": 900000000 + int(i),
                "category": category,
                "actual_label": label,
                "model_proba_at_export": float(proba.loc[i]),
                "features": row,
            })

    OUT_PATH.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Saved {len(examples)} synthetic example transactions to {OUT_PATH}")


if __name__ == "__main__":
    main()
