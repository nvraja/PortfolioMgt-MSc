# src/rl/ppo.py
"""
A compact PPO update loop for policy and value networks using PyTorch.
This is a minimal implementation intended for experimentation (not production).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(64,64)):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        self.net = nn.Sequential(*layers)
        self.actor = nn.Linear(d, output_dim)
        self.critic = nn.Linear(d, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

def choose_action(model: MLPPolicy, obs_vec: np.ndarray):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        logits, value = model(x)
        probs = torch.softmax(logits, dim=-1)
        # for continuous simple approach treat logits as positive weights
        w = probs.squeeze(0).cpu().numpy()
        # renormalize to sum 1
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
    return w, float(value.item())

# Note: full PPO training loop is in trainer.py
