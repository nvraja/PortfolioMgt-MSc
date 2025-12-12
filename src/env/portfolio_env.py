# src/envs/portfolio_env.py
"""Gymnasium-compatible RL environment for portfolio allocation with QP projection.

Key ideas:
- Agent outputs a raw proposal `hat_w` (shape N).
- Environment projects hat_w -> feasible w_star using a QP solved by cvxpy (OSQP).
- Environment simulates one-day step using provided returns; computes reward with costs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Optional
import cvxpy as cp

from ..config import QP_DEFAULTS, BACKTEST_DEFAULTS, EPS

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        price_panel: pd.DataFrame,
        volume_panel: Optional[pd.DataFrame] = None,
        mask_panel: Optional[pd.DataFrame] = None,
        window: int = 1,
        qp_params: Optional[Dict] = None,
        backtest_params: Optional[Dict] = None,
    ):
        """
        price_panel: DataFrame indexed by date, columns=assets with close prices.
        volume_panel: same shape containing volumes (optional).
        mask_panel: same shape boolean True if tradable.
        """
        super().__init__()
        self.dates = price_panel.index.sort_values()
        self.assets = list(price_panel.columns)
        self.N = len(self.assets)
        self.prices = price_panel
        self.volumes = volume_panel if volume_panel is not None else price_panel*0.0 + 1.0
        self.mask = (mask_panel if mask_panel is not None else pd.DataFrame(True, index=self.dates, columns=self.assets))

        self.window = window
        self.qp_params = {**QP_DEFAULTS, **(qp_params or {})}
        self.backtest = {**BACKTEST_DEFAULTS, **(backtest_params or {})}

        # observation = last window of features; here minimal: latest prices & prev weights
        obs_dim = self.N * 2  # price returns + prev weights
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # action: raw proposals (unnormalized). We'll accept continuous N-dim vector and normalize inside env.
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.N,), dtype=np.float32)

        # state
        self._ptr = 0
        self.nav = 1.0
        self.w_prev = np.zeros(self.N)
        self.w_prev.fill(1.0 / max(1, self.N))

    def reset(self, seed=None, options=None):
        self._ptr = self.window
        self.nav = 1.0
        self.w_prev = np.ones(self.N) / self.N
        return self._get_obs(), {}

    def _get_obs(self):
        # returns: [last returns per asset, prev weights]
        # compute last returns (t / t-1)
        t = self._ptr
        prev_close = self.prices.iloc[t - 1].values
        cur_close = self.prices.iloc[t].values
        ret = np.nan_to_num(np.log(cur_close / (prev_close + EPS)))
        obs = np.concatenate([ret, self.w_prev])
        return obs.astype(np.float32)

    def step(self, action):
        """
        action: raw vector hat_w. We'll normalize then project via QP to get feasible w_star.
        """
        hat_w = np.asarray(action, dtype=float)
        # normalize to simplex as a warm-start (can be negative if allowed, we clamp to [0,inf))
        hat_w = np.maximum(hat_w, 0.0)
        if hat_w.sum() <= 0:
            hat_w = np.ones_like(hat_w) / len(hat_w)
        else:
            hat_w = hat_w / hat_w.sum()

        # get market inputs for current timestep
        t = self._ptr
        Sigma = self._estimate_covariance(t)
        mask_row = self.mask.iloc[t].values.astype(bool)

        # project
        w_star = self._qp_project(hat_w, self.w_prev, Sigma, mask_row)

        # get realized returns from t to t+1
        if t + 1 >= len(self.dates):
            done = True
            # no next price, treat as terminal with zero reward
            reward = 0.0
            self._ptr = t + 1
            return self._get_obs(), reward, done, False, {}
        price_t = self.prices.iloc[t].values
        price_t1 = self.prices.iloc[t + 1].values
        simple_ret = (price_t1 - price_t) / (price_t + EPS)
        gross = float(np.dot(w_star, np.nan_to_num(simple_ret)))

        # transaction cost
        delta = w_star - self.w_prev
        tx_cost = self.backtest["fee"] * np.sum(np.abs(delta))

        # drawdown and turnover penalties handled externally or as part of reward shaping
        reward = gross - tx_cost

        # update navigation
        nav_pre = self.nav * (1 + gross)
        self.nav = nav_pre - tx_cost
        self.w_prev = (w_star * (1 + simple_ret)) / (1 + gross + EPS)  # post-return normalization
        self._ptr += 1
        done = (self._ptr >= len(self.dates) - 1)
        return self._get_obs(), float(reward), bool(done), False, {}

    def _estimate_covariance(self, t, window=90):
        """Estimate sample covariance of returns for the last `window` days ending at t. Return NxN matrix."""
        start = max(1, t - window)
        closes = self.prices.iloc[start:t + 1]
        rets = np.log(closes / closes.shift(1)).dropna()
        if len(rets) < 2:
            return np.eye(self.N) * 1e-6
        S = rets.cov().values
        # ensure finite
        S = np.nan_to_num(S, nan=1e-6, posinf=1e-6, neginf=1e-6)
        # pad to N x N if some assets missing
        if S.shape != (self.N, self.N):
            S2 = np.eye(self.N) * 1e-6
            # attempt align
            try:
                cols = rets.columns
                idx_map = [self.assets.index(c) for c in cols]
                for i, ii in enumerate(idx_map):
                    for j, jj in enumerate(idx_map):
                        S2[ii, jj] = S[i, j]
                return S2
            except Exception:
                return np.eye(self.N) * 1e-6
        return S

    def _qp_project(self, hat_w, w_prev, Sigma, mask_row):
        """Solve QP:
            min_w ||w - hat_w||^2 + gamma * w^T Sigma w
            s.t. sum w = 1, 0 <= w_i <= w_max, turnover L1 <= turnover_limit
            mask_row: boolean array where False => enforced w_i = 0
        Uses cvxpy with OSQP backend.
        """
        N = self.N
        w = cp.Variable(N)
        gamma = float(self.qp_params["gamma_risk"])
        w_max = float(self.qp_params["w_max"])
        T_max = float(self.qp_params["turnover_limit"])

        objective = cp.sum_squares(w - hat_w) + gamma * cp.quad_form(w, Sigma + np.eye(N) * 1e-8)
        constraints = []
        constraints.append(cp.sum(w) == 1.0)
        constraints.append(w >= 0.0)
        constraints.append(w <= w_max)
        # mask enforcement
        for i in range(N):
            if not bool(mask_row[i]):
                constraints.append(w[i] == 0.0)
        # turnover linearization
        u = cp.Variable(N, nonneg=True)
        constraints += [
            u >= w - w_prev,
            u >= -(w - w_prev),
            cp.sum(u) <= T_max
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=self.qp_params.get("qp_verbose", False))
            sol = np.array(w.value).astype(float)
            # numeric safety
            sol = np.nan_to_num(sol, nan=0.0)
            # re-normalize to sum to 1 if tiny rounding issues
            su = sol.sum()
            if su <= 0:
                # fallback to uniform over allowed assets
                allowed = (~np.isnan(sol)) & (mask_row) & (np.isfinite(sol))
                if allowed.sum() == 0:
                    sol = np.ones(N) / N
                else:
                    sol = np.where(allowed, 1.0 / allowed.sum(), 0.0)
            else:
                sol = sol / su
            return sol
        except Exception as e:
            # on solver failure, fallback to simple clipped normalization of hat_w
            w0 = np.maximum(hat_w, 0.0)
            if w0.sum() <= 0:
                w0 = np.ones(N) / N
            else:
                w0 = w0 / w0.sum()
            w0 = np.minimum(w0, w_max)
            if w0.sum() <= 0:
                w0 = np.ones(N) / N
            return w0

    def render(self):
        print(f"Date: {self.dates[self._ptr-1] if self._ptr>0 else 'NA'} NAV: {self.nav:.4f}")

