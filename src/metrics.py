"""Regression metrics computation.

Provides a single :func:`evaluate` helper returning MAE, MSE, RMSE and R²
for a set of predictions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with keys ``mae``, ``mse``, ``rmse``, ``r2``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
