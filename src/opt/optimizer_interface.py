# src/opt/optimizer_interface.py
"""
High-level optimizer interface.

Provides:
- project_weights(hat_w, w_prev, Sigma, mask, qp_params) -> feasible weights
- solve_qp_matrix(P, q, A, l, u, warm_start=None) -> primal solution (uses OSQP)
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

from .solvers import solve_qp_cvxpy, solve_qp_osqp_matrix
from ..config import QP_DEFAULTS

log = logging.getLogger(__name__)

def project_weights(
    hat_w: np.ndarray,
    w_prev: np.ndarray,
    Sigma: np.ndarray,
    mask: Optional[np.ndarray] = None,
    qp_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Project `hat_w` onto feasible set via a QP:
      min_w ||w - hat_w||^2 + gamma * w^T Sigma w
      s.t. sum(w) == 1, 0 <= w_i <= w_max, turnover L1 <= turnover_limit, masked assets = 0

    This function tries the cvxpy solver first (clean readable form) and falls back to a fast OSQP matrix builder on failure.
    """
    params = {**QP_DEFAULTS, **(qp_params or {})}
    N = len(hat_w)
    mask = np.ones(N, dtype=bool) if mask is None else mask.astype(bool)

    # Build problem data for cvxpy
    try:
        sol = solve_qp_cvxpy(hat_w, w_prev, Sigma, mask, params)
        if sol is not None and np.isfinite(sol).all():
            return sol
    except Exception as exc:
        log.warning("cvxpy QP failed: %s — will try OSQP matrix fallback", exc)

    # fallback: matrix formulation for OSQP
    try:
        sol = solve_qp_osqp_matrix(hat_w, w_prev, Sigma, mask, params)
        if sol is not None and np.isfinite(sol).all():
            return sol
    except Exception as exc:
        log.error("OSQP matrix solver failed: %s. Falling back to clipped normalized hat_w", exc)

    # final fallback: clipped normalized hat_w
    w0 = np.maximum(hat_w, 0.0)
    s = w0.sum()
    if s <= 0:
        # uniform on allowed assets
        allowed = mask
        if allowed.sum() == 0:
            return np.ones(N) / N
        out = np.where(allowed, 1.0 / allowed.sum(), 0.0)
        return out
    w0 = w0 / s
    w0 = np.minimum(w0, params["w_max"])
    w0 = w0 / w0.sum()
    return w0
