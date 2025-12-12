# src/backtest/backtester.py
"""
Backtester: simulate NAV given sequences of proposed allocations (or an RL agent + env).
Provides `run_backtest` which accepts either:
- an agent with .propose(state) or
- a sequence of w_t decisions
and produces NAV, weights history, and transaction costs.
"""

from typing import Callable, Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, max_drawdown, cvar
from ..config import BACKTEST_DEFAULTS

def run_backtest_from_weights(
    prices: pd.DataFrame,
    weight_seq: List[np.ndarray],
    fee: float = BACKTEST_DEFAULTS["fee"],
    slippage_kappa: float = BACKTEST_DEFAULTS["slippage_kappa"],
    initial_nav: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    prices: DataFrame indexed by date columns=assets
    weight_seq: list/array of N-d weight vectors, length = len(prices)-1 (weights used at close t to next day)
    returns output dict with nav_history, pnl_series, turnover_series, tx_costs
    """
    dates = prices.index
    N = prices.shape[1]
    nav = float(initial_nav)
    nav_history = [nav]
    weights = []
    tx_costs = []
    turnovers = []
    pnl_series = []

    w_prev = np.zeros(N)
    w_prev[:] = 0.0
    # if first weight not provided, use uniform
    if len(weight_seq) == 0:
        raise ValueError("weight_seq must be non-empty")
    for t, w in enumerate(weight_seq):
        # ensure array
        w = np.asarray(w, dtype=float)
        # normalize to sum 1
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = np.maximum(w, 0.0)
            w = w / w.sum()
        # compute return from day t to t+1
        if t + 1 >= len(dates):
            break
        p_t = prices.iloc[t].values
        p_t1 = prices.iloc[t + 1].values
        simple_ret = (p_t1 - p_t) / (p_t + 1e-12)
        # gross pnl
        gross_ret = float(np.dot(w, simple_ret))
        # transaction cost (simple proportional + slippage quadratic approx)
        delta = np.abs(w - w_prev)
        tx = fee * delta.sum() + slippage_kappa * np.sum((delta * nav)**2 / (np.maximum(1.0, prices.iloc[t].values)))
        nav_pre = nav * (1 + gross_ret)
        nav_post = nav_pre - tx
        pnl_series.append(nav_post - nav)
        nav = nav_post
        nav_history.append(nav)
        weights.append(w.copy())
        tx_costs.append(tx)
        turnovers.append(delta.sum())
        w_prev = (w * (1 + simple_ret)) / (1 + gross_ret + 1e-12)
    # compute metrics
    pnl_arr = np.array(pnl_series)
    metrics = {
        "nav_history": np.array(nav_history),
        "pnl_series": pnl_arr,
        "tx_costs": np.array(tx_costs),
        "turnovers": np.array(turnovers),
        "sharpe": sharpe_ratio(pnl_arr),
        "max_drawdown": max_drawdown(np.array(nav_history)),
        "cvar_95": cvar(pnl_arr, 0.95),
    }
    return metrics

def run_backtest_with_agent(
    env,
    agent,
    max_steps: Optional[int] = None,
    render: bool = False
) -> Dict[str, Any]:
    """
    Run an episode in the provided environment using `agent.propose(obs)` interface.
    env should be compatible with the Gym env used earlier (reset() -> obs, step(action)).
    Agent must implement propose(obs) -> action (numpy array).
    """
    obs, _ = env.reset()
    done = False
    nav_history = []
    weights = []
    tx_costs = []
    rewards = []
    step = 0
    while not done:
        action = agent.propose(obs)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        # environment stores nav in env.nav and w_prev in env.w_prev
        nav_history.append(env.nav)
        tx_costs.append(info.get("tx_cost", 0.0) if isinstance(info, dict) else 0.0)
        weights.append(env.w_prev.copy())
        if render:
            env.render()
        step += 1
        if max_steps and step >= max_steps:
            break
    rewards = np.array(rewards)
    metrics = {
        "nav_history": np.array(nav_history),
        "rewards": rewards,
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "sharpe": sharpe_ratio(rewards),
        "max_drawdown": max_drawdown(np.array(nav_history)),
    }
    return metrics
