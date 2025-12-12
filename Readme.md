# Hybrid-RL-Crypto-Portfolio

Hybrid RL + convex-optimization framework for short-term cryptocurrency portfolio management.

**Key idea:** an RL policy (PPO) proposes allocations which are projected to a feasible, risk-aware portfolio by a QP layer (OSQP/CVXPY). Includes reproducible backtests, walk-forward evaluation and overfitting rejection tests.

## Repo structure
project-root/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ environment.yml
├─ setup.py
├─ data/
│ └─ raw/
├─ notebooks/
├─ src/
│ ├─ data/
│ ├─ envs/
│ ├─ rl/
│ ├─ opt/
│ └─ backtest/
├─ experiments/
└─ tests/


## Quick start (development)
```bash
# 1) New conda env 
conda env create -f environment.yml
conda activate hybrid-rl

# 2) If using pip
pip install -e .

# 3) Run tests
pytest -q

# Home to run the demo 
python demo.py               

# Data
data/raw/
