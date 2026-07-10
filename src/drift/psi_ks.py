"""Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift statistics.

PSI is the standard industry metric for comparing a "reference" distribution (e.g. the
training window) against a "current" one (e.g. a later time window) — it's bucketed, so
it works the same way for numeric and categorical features. Conventional thresholds:
< 0.1 no significant shift, 0.1-0.2 moderate shift worth watching, > 0.2 significant
shift that should trigger investigation/retraining.

KS is a numeric-only, threshold-free complement: it's the max distance between the two
empirical CDFs plus a p-value, so it flags shift even when PSI's binning smooths it out.
"""

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def compute_psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).values)
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = pd.cut(reference, bins=edges).value_counts(sort=False)
    cur_counts = pd.cut(current, bins=edges).value_counts(sort=False)

    ref_pct = (ref_counts / ref_counts.sum()).clip(lower=EPS)
    cur_pct = (cur_counts / cur_counts.sum()).clip(lower=EPS)

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def compute_psi_categorical(reference: pd.Series, current: pd.Series, max_categories: int = 20) -> float:
    reference = reference.astype("object").fillna("__missing__")
    current = current.astype("object").fillna("__missing__")

    top_categories = reference.value_counts().head(max_categories).index
    ref_grouped = reference.where(reference.isin(top_categories), "__other__")
    cur_grouped = current.where(current.isin(top_categories), "__other__")

    categories = sorted(set(ref_grouped.unique()) | set(cur_grouped.unique()))
    ref_pct = ref_grouped.value_counts(normalize=True).reindex(categories, fill_value=0).clip(lower=EPS)
    cur_pct = cur_grouped.value_counts(normalize=True).reindex(categories, fill_value=0).clip(lower=EPS)

    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def compute_ks(reference: pd.Series, current: pd.Series) -> tuple:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) < 2 or len(current) < 2:
        return np.nan, np.nan
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def psi_severity(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    if psi < 0.1:
        return "none"
    if psi < 0.2:
        return "moderate"
    return "significant"


def compute_drift_table(reference: pd.DataFrame, current: pd.DataFrame,
                         numeric: list, categorical: list) -> pd.DataFrame:
    rows = []
    for col in numeric:
        psi = compute_psi_numeric(reference[col], current[col])
        ks_stat, ks_pvalue = compute_ks(reference[col], current[col])
        rows.append({"feature": col, "type": "numeric", "psi": psi,
                      "ks_stat": ks_stat, "ks_pvalue": ks_pvalue,
                      "severity": psi_severity(psi)})
    for col in categorical:
        psi = compute_psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi,
                      "ks_stat": np.nan, "ks_pvalue": np.nan,
                      "severity": psi_severity(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
