"""Shared preprocessing pipeline for models that need dense, imputed, scaled input
(Logistic Regression). Tree models (LightGBM/XGBoost) consume raw features directly and
don't use this — they handle missing values and categoricals natively.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ["TransactionID", "isFraud", "TransactionDT"]


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    categorical = [c for c in feature_cols if df[c].dtype == object]
    numeric = [c for c in feature_cols if c not in categorical]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric, categorical = get_feature_columns(df)

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, max_categories=20)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical),
    ])
    return preprocessor, numeric, categorical
