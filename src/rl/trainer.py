# src/rl/trainer.py
"""
Very small trainer that performs random policy iterations for a few epochs.
This is a placeholder to show how training can be orchestrated.
"""
import torch
import numpy as np
from .ppo import MLPPolicy
from .agent import PolicyAgent
from ..utils.logging import get_logger

log = get_logger(__name__)

def build_policy_for_env(env):
    # observation dim: prices + w_prev (same length = n_assets each)
    input_dim = env.n_assets * 2
    output_dim = env.n_assets
    model = MLPPolicy(input_dim, output_dim)
    return model

def quick_train(env, epochs=5, lr=3e-4):
    model = build_policy_for_env(env)
    agent = PolicyAgent(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Very small random policy gradient-ish loop: collect rollout using current policy, compute returns, and do a simple update on value
    for ep in range(epochs):
        obs, _ = env.reset()
        done = False
        losses = []
        while not done:
            w = agent.propose(obs)
            obs_next, reward, done, _, info = env.step(w)
            # tiny supervised target: encourage weights that gave positive reward (toy)
            loss = -reward  # we want to maximize reward so minimize -reward
            losses.append(loss)
            obs = obs_next
        if losses:
            L = np.mean(losses)
            opt.zero_grad()
            # pretend L is tensor
            torch.tensor(L, requires_grad=True).backward()
            opt.step()
        log.info("Epoch %d finished epoch_loss=%.6f nav=%.2f", ep, L if losses else 0.0, env.nav)
    return model
