from __future__ import annotations
import numpy as np

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-6)