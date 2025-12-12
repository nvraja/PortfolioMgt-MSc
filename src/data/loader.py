# src/data/loader.py
"""Simple OHLCV dataset loader. Supports CSV and Parquet snapshots.

Functions:
- load_ohlcv: load (date, asset, o,h,l,c,v) into a MultiIndex DataFrame or panel-like dict.
- sample_data: small helper to produce a tiny synthetic dataset for demos/tests.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Iterable, Dict

def load_ohlcv(path: Path, assets: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load OHLCV from a CSV or Parquet file into a tidy DataFrame with columns:
    ['date', 'asset', 'open','high','low','close','volume'].

    Expects either a file with those columns or a parquet partitioned dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif path.suffix in [".csv", ".gz"]:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        # treat as directory of parquet/CSV files
        files = list(path.glob("*.parquet")) + list(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No parquet/csv files under {path}")
        dfs = [pd.read_parquet(f) if f.suffix==".parquet" else pd.read_csv(f, parse_dates=["date"]) for f in files]
        df = pd.concat(dfs, ignore_index=True)

    # canonicalize columns
    expected = {"date", "asset", "open", "high", "low", "close", "volume"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Input file missing columns. Found: {df.columns.tolist()}")

    df = df.loc[:, list(expected)].copy()
    df["date"] = pd.to_datetime(df["date"])
    if assets is not None:
        df = df[df["asset"].isin(set(assets))].copy()

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    return df

def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with dates as index and assets as columns containing close prices."""
    return df.pivot(index="date", columns="asset", values="close").sort_index()

def sample_data(n_assets=4, n_days=300, seed=0) -> pd.DataFrame:
    """Generate small synthetic OHLCV dataset for demos and unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    assets = [f"ASSET{i+1}" for i in range(n_assets)]
    rows = []
    for a in assets:
        price = 1.0 + 0.001 * rng.standard_normal()  # start
        for d in dates:
            ret = rng.normal(0, 0.02)
            price = price * (1 + ret)
            o = price * (1 + rng.normal(0, 0.002))
            h = o * (1 + abs(rng.normal(0, 0.01)))
            l = o * (1 - abs(rng.normal(0, 0.01)))
            c = price
            v = max(1e3, rng.integers(1000, 1_000_000))
            rows.append({"date": d, "asset": a, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return pd.DataFrame(rows)
