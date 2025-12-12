# Hybrid-RL-Crypto-Portfolio

Hybrid RL + convex-optimization framework for short-term cryptocurrency portfolio management.

**Key idea:** an RL policy (PPO) proposes allocations which are projected to a feasible, risk-aware portfolio by a QP layer (OSQP/CVXPY). Includes reproducible backtests, walk-forward evaluation and overfitting rejection tests.

## Repository structure

- **project-root/**
  - `README.md`
  - `LICENSE`
  - `requirements.txt`
  - `environment.yml`
  - `setup.py`
  - **data/**
    - `raw/`
  - **notebooks/**
  - **src/**
    - **data/**
      - `loader.py`
      - `clean.py`
      - `features.py`
    - **envs/**
      - `portfolio_env.py`
    - **rl/**
      - `ppo.py`
      - `agent.py`
      - `trainer.py`
    - **opt/**
      - `optimizer_interface.py`
      - `solvers.py`
    - **backtest/**
      - `backtester.py`
  - **experiments/**
  - **tests/**


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
