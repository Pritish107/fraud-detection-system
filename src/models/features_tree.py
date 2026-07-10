"""Feature prep for tree models (LightGBM). Unlike the Logistic Regression baseline,
LightGBM handles missing values and categorical splits natively, so no imputation or
one-hot encoding is needed — just typing object columns as pandas 'category' dtype so
LightGBM treats them as categorical rather than trying to parse them as numbers.
"""

from typing import List, Tuple

import pandas as pd

from models.preprocessing import ID_COLS, get_feature_columns


def prepare_tree_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric, categorical = get_feature_columns(df)
    df = df.copy()
    for c in categorical:
        df[c] = df[c].astype("category")
    feature_cols = numeric + categorical
    return df[feature_cols + [c for c in ID_COLS if c in df.columns]], numeric, categorical
