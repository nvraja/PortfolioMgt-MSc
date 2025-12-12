# src/data/clean.py
"""Cleaning helpers: small-gap forward fill, masking low-volume days, delisting handling."""

import pandas as pd
import numpy as np
from typing import Tuple

def forward_fill_small_gaps(df: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """
    For each asset, forward-fill price columns for gaps of length <= max_gap.
    Assumes DataFrame sorted by ['asset','date'] and contains OHLCV columns.
    """
    out = []
    for asset, g in df.groupby("asset", sort=False):
        g = g.copy().set_index("date").sort_index()
        # Count consecutive NaNs in 'close'
        is_na = g["close"].isna()
        if is_na.any():
            # fill only small runs
            runs = (is_na != is_na.shift()).cumsum()
            run_lens = is_na.groupby(runs).transform("sum")
            small = (is_na) & (run_lens <= max_gap)
            g.loc[small, ["open", "high", "low", "close"]] = g[["open","high","low","close"]].ffill().loc[small]
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True).sort_values(["asset","date"])

def make_mask(df: pd.DataFrame, vol_eps: float = 1e-6) -> pd.DataFrame:
    """
    Create a mask column per row indicating whether the day is tradable for that asset.
    Returns df with a new boolean column 'mask' (True=tradable).
    Criteria: volume > vol_eps and close != 0 and notna.
    """
    df = df.copy()
    df["mask"] = (~df["close"].isna()) & (df["close"] != 0) & (df["volume"] > vol_eps)
    return df

def apply_delisting_mask(df: pd.DataFrame, min_history_days: int = 30) -> pd.DataFrame:
    """
    For assets with short history (< min_history_days), mask them out for those early days.
    This is a simple policy: require at least min_history_days of consecutive non-missing values before enabling allocations.
    """
    out = []
    for asset, g in df.groupby("asset", sort=False):
        g = g.copy().sort_values("date")
        valid = (~g["close"].isna()) & (g["close"] != 0) & (g["volume"] > 0)
        # rolling count of valid days
        running = valid.cumsum()
        g["mask"] = g.get("mask", True) & (running >= min_history_days)
        out.append(g)
    return pd.concat(out, ignore_index=True).sort_values(["asset","date"])
