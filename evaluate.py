"""Regression + business metrics for ETA evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def within_pct(y_true, y_pred, pct):
    """Share of predictions within `pct` of the actual value — the operational
    accuracy metric the brief asks for (within 15%)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rel = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(rel <= pct) * 100.0)


def score(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "within_10pct": within_pct(y_true, y_pred, 0.10),
        "within_15pct": within_pct(y_true, y_pred, 0.15),
    }
