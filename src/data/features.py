# src/data/features.py
"""Feature engineering: rolling returns, realized volatility, momentum, normalized z-scores."""

import pandas as pd
import numpy as np
from typing import List

def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """close: DataFrame indexed by date columns=assets -> returns DataFrame of log returns."""
    return np.log(close / close.shift(1)).fillna(0.0)

def rolling_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Realized volatility (std of returns) on rolling window."""
    return returns.rolling(window).std().fillna(method="bfill")

def momentum(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Simple momentum: price / price_{t-window} - 1"""
    return close / close.shift(window) - 1.0

def rolling_zscore(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Per-asset rolling z-score normalization."""
    mu = df.rolling(window).mean()
    sigma = df.rolling(window).std().replace(0, np.nan)
    z = (df - mu) / sigma
    return z.fillna(0.0)

def build_feature_tensor(close: pd.DataFrame, volume: pd.DataFrame, windows=(5,20,60)) -> pd.DataFrame:
    """
    Construct a feature DataFrame indexed by date with MultiIndex columns (asset, feature).
    Features: returns, vol(20), momentum(20), MA ratios for windows param.
    """
    returns = compute_log_returns(close)
    vol20 = rolling_volatility(returns, window=20)
    mom20 = momentum(close, window=20)
    features = []
    for asset in close.columns:
        df_asset = pd.DataFrame(index=close.index)
        df_asset["ret_1"] = returns[asset]
        df_asset["vol_20"] = vol20[asset]
        df_asset["mom_20"] = mom20[asset]
        # MA ratios
        for w in windows:
            ma_short = close[asset].rolling(window=max(1,int(w/2))).mean()
            ma_long = close[asset].rolling(window=w).mean()
            df_asset[f"ma_ratio_{w}"] = (ma_short / (ma_long + 1e-12)) - 1.0
        # volume change
        df_asset["vol_30chg"] = (volume[asset].rolling(30).mean() / (volume[asset].rolling(60).mean() + 1e-12)) - 1.0
        # prefix columns by asset in returned stacked form
        df_asset.columns = pd.MultiIndex.from_product([[asset], df_asset.columns])
        features.append(df_asset)
    # concat along columns
    feat = pd.concat(features, axis=1)
    # optional: compress to single-level with asset|feature names
    feat.columns = ["|".join(col) for col in feat.columns]
    return feat
