# tests/test_optimizer.py
import numpy as np
from src.opt.optimizer_interface import project_weights

def test_project_simple():
    N = 3
    hat = np.array([0.8, 0.1, 0.1])
    w_prev = np.ones(N)/N
    Sigma = np.eye(N)*0.01
    out = project_weights(hat, w_prev, Sigma, mask=np.ones(N,dtype=bool), qp_params={"w_max":0.6})
    assert out.shape == (N,)
    assert abs(out.sum() - 1.0) < 1e-6
    assert (out >= 0).all()
