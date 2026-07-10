"""SHAP-based explanations for the main LightGBM fraud model.

Used both by the EDA/report generation script (global + example local explanations)
and by the FastAPI service (per-request local explanation of "why was this flagged").
"""

import json
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


class FraudExplainer:
    def __init__(self, model_path: Optional[Path] = None, meta_path: Optional[Path] = None):
        model_path = model_path or MODELS_DIR / "main_model.txt"
        meta_path = meta_path or MODELS_DIR / "main_model_meta.json"

        self.booster = lgb.Booster(model_file=str(model_path))
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.numeric_features: List[str] = self.meta["numeric_features"]
        self.categorical_features: List[str] = self.meta["categorical_features"]
        self.feature_cols: List[str] = self.numeric_features + self.categorical_features
        self.threshold: float = self.meta["threshold"]

        self._explainer = shap.TreeExplainer(self.booster)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.numeric_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.categorical_features:
            if c in df.columns:
                df[c] = df[c].astype("object").where(df[c].notna(), None).astype("category")
        return df[self.feature_cols]

    def predict_proba(self, df: pd.DataFrame) -> "pd.Series[float]":
        X = self._prepare(df)
        return pd.Series(self.booster.predict(X), index=df.index)

    def shap_values(self, df: pd.DataFrame):
        X = self._prepare(df)
        return self._explainer.shap_values(X), X

    def explain_row(self, row: pd.DataFrame, top_n: int = 5) -> dict:
        """Explain a single-row DataFrame. Returns probability, decision, and the
        top_n features driving the prediction (toward or away from fraud)."""
        X = self._prepare(row)
        proba = float(self.booster.predict(X)[0])
        sv = self._explainer.shap_values(X)[0]

        contributions = sorted(
            zip(self.feature_cols, sv, X.iloc[0].tolist()),
            key=lambda t: abs(t[1]), reverse=True,
        )[:top_n]

        return {
            "fraud_probability": proba,
            "decision": "fraud" if proba >= self.threshold else "not_fraud",
            "threshold": self.threshold,
            "top_features": [
                {"feature": f, "shap_value": float(v), "feature_value": _clean(val)}
                for f, v, val in contributions
            ],
        }


def _clean(val):
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val
