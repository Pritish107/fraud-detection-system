"""Time-based train/validation/test split.

Fraud rate drifts over the observed window (see reports/eda/eda_report.md), and in
production the model always predicts on transactions that happened after training data
was collected. A random split would leak future information into training and overstate
performance, so we sort by TransactionDT and split chronologically instead.
"""

import pandas as pd


def time_based_split(df: pd.DataFrame, dt_col: str = "TransactionDT",
                      train_frac: float = 0.70, val_frac: float = 0.15):
    df_sorted = df.sort_values(dt_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test
