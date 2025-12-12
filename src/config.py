# src/config.py
"""Global configuration and defaults used across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data locations (override in experiments config)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# QP / optimizer defaults
QP_DEFAULTS = {
    "gamma_risk": 1.0,        # weight on w^T Sigma w
    "w_max": 0.30,            # per-asset cap
    "turnover_limit": 0.5,    # L1 norm maximum change
    "solver": "OSQP",         # backend for cvxpy
    "qp_verbose": False,
}

# RL / training defaults
RL_DEFAULTS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "lr": 3e-4,
    "batch_size": 64,
}

# Backtest defaults
BACKTEST_DEFAULTS = {
    "fee": 0.001,    # 0.1%
    "slippage_kappa": 0.2,
}

# Some small numeric epsilon
EPS = 1e-9
