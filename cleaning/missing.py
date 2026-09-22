"""Utilities for handling missing values in pandas DataFrames."""

import pandas as pd
from typing import List, Optional


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def fill_missing(df: pd.DataFrame, strategy: str = "median",
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fill missing values using the given strategy: mean, median, mode, or zero."""
    df = df.copy()
    cols = columns or df.select_dtypes(include="number").columns.tolist()

    for col in cols:
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
    return df
