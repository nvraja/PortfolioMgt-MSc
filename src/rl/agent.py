# src/rl/agent.py
"""
Simple agent wrapper around policy network. Exposes propose(obs) used by backtester/env.
"""
import numpy as np
import torch
from .ppo import MLPPolicy

class RandomAgent:
    def __init__(self, n_assets, seed=0):
        self.n_assets = n_assets
        np.random.seed(seed)

    def propose(self, obs):
        v = np.random.rand(self.n_assets)
        v = np.maximum(v, 0.0)
        return v / v.sum()

class PolicyAgent:
    def __init__(self, model: MLPPolicy, device="cpu"):
        self.model = model
        self.device = device

    def propose(self, obs):
        # obs expected to be dict {"prices": arr, "w_prev": arr}
        # build simple feature vector: normalized prices + w_prev
        p = obs["prices"].astype(np.float32)
        p = (p - p.mean()) / (p.std() + 1e-8)
        wprev = obs["w_prev"].astype(np.float32)
        vec = np.concatenate([p, wprev])
        from .ppo import choose_action
        w, _ = choose_action(self.model, vec)
        return w
