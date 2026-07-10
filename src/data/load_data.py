"""Load and merge the IEEE-CIS transaction/identity tables with a reduced memory footprint.

The raw CSVs are ~1.3GB combined and pandas' default dtype inference (float64/int64
for almost every column) roughly doubles that in memory. `reduce_memory_usage` downcasts
each column to the smallest dtype that can hold its values without loss.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mb = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == object:  # noqa: E721 -- standard pandas dtype-comparison idiom
            continue

        c_min, c_max = df[col].min(), df[col].max()
        if pd.isna(c_min) or pd.isna(c_max):
            continue

        if np.issubdtype(col_dtype, np.integer):
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                info = np.iinfo(dtype)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            for dtype in (np.float32, np.float64):
                info = np.finfo(dtype)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break

    end_mb = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory usage: {start_mb:.1f} MB -> {end_mb:.1f} MB "
              f"({100 * (start_mb - end_mb) / start_mb:.0f}% reduction)")
    return df


def load_transaction(split: str = "train", nrows: Optional[int] = None) -> pd.DataFrame:
    path = RAW_DIR / f"{split}_transaction.csv"
    df = pd.read_csv(path, nrows=nrows)
    return reduce_memory_usage(df, verbose=False)


def load_identity(split: str = "train", nrows: Optional[int] = None) -> pd.DataFrame:
    path = RAW_DIR / f"{split}_identity.csv"
    df = pd.read_csv(path, nrows=nrows)
    return reduce_memory_usage(df, verbose=False)


def load_merged(split: str = "train", nrows: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    """Left-join transaction with identity on TransactionID (most transactions have no identity row)."""
    tx = load_transaction(split, nrows=nrows)
    idn = load_identity(split)
    merged = tx.merge(idn, on="TransactionID", how="left")
    if verbose:
        reduce_memory_usage(merged, verbose=True)
    return merged


if __name__ == "__main__":
    df = load_merged("train")
    print(df.shape)
