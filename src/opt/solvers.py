# src/opt/solvers.py
"""
Solver helpers.
- solve_qp_cvxpy: builds readable cvxpy problem (same objective as env).
- solve_qp_osqp_matrix: constructs OSQP matrices (P, q, A, l, u) and calls OSQP directly for speed.
"""

import numpy as np
import cvxpy as cp
import osqp
import scipy.sparse as sp
from typing import Optional, Dict, Any

from ..config import EPS

def solve_qp_cvxpy(hat_w, w_prev, Sigma, mask, params: Optional[Dict[str,Any]] = None):
    params = params or {}
    gamma = float(params.get("gamma_risk", 1.0))
    w_max = float(params.get("w_max", 0.3))
    T_max = float(params.get("turnover_limit", 0.5))
    qp_verbose = bool(params.get("qp_verbose", False))

    N = len(hat_w)
    w = cp.Variable(N)
    u = cp.Variable(N, nonneg=True)

    objective = cp.sum_squares(w - hat_w) + gamma * cp.quad_form(w, Sigma + np.eye(N)*1e-8)
    constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= w_max]
    for i in range(N):
        if not bool(mask[i]):
            constraints.append(w[i] == 0.0)
    constraints += [u >= w - w_prev, u >= -(w - w_prev), cp.sum(u) <= T_max]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=qp_verbose)
    if prob.status not in {"optimal", "optimal_inaccurate"}:
        return None
    sol = np.array(w.value).astype(float)
    sol = np.nan_to_num(sol, nan=0.0)
    s = sol.sum()
    if s <= 0:
        # uniform over allowed
        allowed = (mask) & (np.isfinite(sol))
        if allowed.sum() == 0:
            return np.ones(N)/N
        sol = np.where(allowed, 1.0/allowed.sum(), 0.0)
    else:
        sol = sol / s
    return sol

def solve_qp_osqp_matrix(hat_w, w_prev, Sigma, mask, params: Optional[Dict[str,Any]] = None):
    params = params or {}
    gamma = float(params.get("gamma_risk", 1.0))
    w_max = float(params.get("w_max", 0.3))
    T_max = float(params.get("turnover_limit", 0.5))
    eps = 1e-8

    N = len(hat_w)
    # objective: 1/2 x^T P x + q^T x  (OSQP uses 1/2 x^T P x + q^T x)
    # Our objective: ||w - hat_w||^2 + gamma * w^T Sigma w
    # => expand: w^T(I + gamma*Sigma)w - 2 hat_w^T w + const
    P = 2.0 * (np.eye(N) + gamma * Sigma + eps * np.eye(N))
    q = -2.0 * hat_w

    # Constraints:
    # 1) sum(w) == 1   -> equality
    # 2) 0 <= w <= w_max
    # 3) masked w_i == 0
    # 4) turnover L1 <= T_max  -> introduce u >= w - w_prev; u >= -(w - w_prev); sum(u) <= T_max
    # We'll build A x <= u form with equality stacked.
    # Variables: x = [w (N), u (N)]
    P_big = sp.csc_matrix(sp.block_diag([P, sp.csc_matrix((N,N))]))
    q_big = np.concatenate([q, np.zeros(N)])

    # Build linear constraints
    # 1) sum(w) == 1 ->  [1...1 | 0...0] eq
    A_eq = np.zeros((1, 2*N))
    A_eq[0, :N] = 1.0
    l_eq = np.array([1.0])
    u_eq = np.array([1.0])

    # 2) bounds: 0 <= w <= w_max -> inequalities
    # 3) mask: w_i == 0  -> equality row per masked index
    ineq_rows = []
    ineq_l = []
    ineq_u = []

    # w bounds
    A_w_lb = sp.eye(N, format="csc")
    A_w_ub = sp.eye(N, format="csc")
    # convert to dense rows below by stacking later

    # turnover constraints:
    # u >= w - w_prev  ->  -w + u >= -w_prev
    # u >= -(w - w_prev) ->  w + u >= w_prev
    tol = 1e-9
    rows = []
    rhs_l = []
    rhs_u = []

    # stack constraints as A x >= l and <= u; OSQP uses l <= A x <= u
    # We'll collect A_rows in a sparse list and stack

    # To simplify: build dense A then convert to sparse (N small typically)
    # Number of constraints:
    # - w lower bounds: N
    # - w upper bounds: N
    # - masked equalities: M
    # - turnover inequalities: 2N
    # Build A matrix
    n_mask_eq = int(np.sum(~mask))
    n_rows = N + N + n_mask_eq + 2*N
    A = np.zeros((n_rows + 1, 2*N))  # +1 for equality sum(w)=1 already added
    row_idx = 1  # we reserved row 0 for equality
    # w >= 0
    A[row_idx:row_idx+N, :N] = np.eye(N)
    l = [-0.0]*N
    u = [w_max]*N  # we'll interpret these as bounds for w (we will set l=0,u=w_max)
    # But OSQP expects l <= Ax <= u; we'll set corresponding l,u below after constructing full A
    # For now create arrays for l_full,u_full
    l_full = []
    u_full = []
    # w lower (0) and upper (w_max)
    for i in range(N):
        l_full.append(0.0)
        u_full.append(w_max)
    row_idx += N

    # masked equality rows
    for i in range(N):
        if not mask[i]:
            A[row_idx, i] = 1.0
            l_full.append(0.0)
            u_full.append(0.0)
            row_idx += 1

    # turnover: -w + u >= -w_prev  and  w + u >= w_prev
    # build u variable index offset N
    for i in range(N):
        # -w + u >= -w_prev  -> row
        A[row_idx, i] = -1.0
        A[row_idx, N + i] = 1.0
        l_full.append(-w_prev[i])
        u_full.append(np.inf)
        row_idx += 1
    for i in range(N):
        A[row_idx, i] = 1.0
        A[row_idx, N + i] = 1.0
        l_full.append(w_prev[i])
        u_full.append(np.inf)
        row_idx += 1

    # sum(u) <= T_max  -> 0..N columns
    A = np.vstack([A_eq, A])
    # final row for sum(u) <= T_max
    last_row = np.zeros((1, 2*N))
    last_row[0, N:] = 1.0
    A = np.vstack([A, last_row])
    l_full = np.concatenate([l_eq, np.array(l_full), [-np.inf]])
    u_full = np.concatenate([u_eq, np.array(u_full), [T_max]])

    A_sparse = sp.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P_big, q_big, A_sparse, l=l_full, u=u_full, verbose=False)
    res = prob.solve()
    if res.info.status_val not in (1, 2):  # 1=solved, 2=solved_inaccurate
        return None
    x = res.x[:N]
    x = np.nan_to_num(x, nan=0.0)
    s = x.sum()
    if s <= 0:
        # fallback uniform over allowed
        allowed = mask
        if allowed.sum() == 0:
            return np.ones(N) / N
        return np.where(allowed, 1.0 / allowed.sum(), 0.0)
    return x / s
