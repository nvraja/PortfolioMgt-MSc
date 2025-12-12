# src/backtest/metrics.py
"""Standard performance metrics."""

import numpy as np
from typing import Sequence

def sharpe_ratio(returns: Sequence[float], freq: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio from per-period returns (not log-returns).
    freq: periods per year (daily->252).
    rf: risk-free rate per period (not annual).
    """
    r = np.array(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = r.mean()
    sigma = r.std(ddof=1) if r.size > 1 else 0.0
    if sigma <= 0:
        return 0.0
    ann_sharpe = (mu - rf) / sigma * np.sqrt(freq)
    return float(ann_sharpe)

def max_drawdown(nav: Sequence[float]) -> float:
    arr = np.array(nav, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (peak + 1e-12)
    return float(np.min(dd))

def cvar(returns: Sequence[float], alpha: float = 0.95) -> float:
    """Empirical CVaR (Expected Shortfall) at level alpha for returns (loss-negative sign not applied)."""
    r = np.array(returns, dtype=float)
    if r.size == 0:
        return 0.0
    # For losses, we compute left tail: sort ascending
    sorted_r = np.sort(r)
    cutoff = int(np.floor((1.0 - alpha) * len(r)))
    if cutoff < 1:
        cutoff = 1
    tail = sorted_r[:cutoff]
    return float(np.mean(tail)) if tail.size else float(sorted_r[0])
