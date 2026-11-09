"""Model training orchestration.

Splits data, scales features, tunes every registered model, builds a
stacking ensemble, selects the best model by RMSE, and persists a full
scikit-learn pipeline (preprocessor + regressor) to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import settings
from exceptions import ModelNotTrainedError
from logger import get_logger
from base import (
    ModelRegistry,
    build_preprocessor,
    build_stacker,
    save_pipeline,
    tune_model,
)
from metrics import evaluate

logger = get_logger(__name__)


def split_data(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature dataframe into train/test matrices.

    Args:
        features: Train-level feature dataframe.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = features[settings.numeric_features]
    y = features[settings.target]
    return train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )


def train_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, dict], str]:
    """Train and evaluate every registered model plus a stacker.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        ``(best_pipeline, metrics_dict, best_model_name)``.
    """
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    registry = ModelRegistry()
    trained: Dict[str, object] = {}
    results: Dict[str, dict] = {}

    # Train linear family directly (no tuning needed / small search).
    for name in ["linear", "ridge", "lasso", "elasticnet"]:
        spec = registry.get(name)
        estimator = tune_model(spec, X_train_scaled, y_train.values)
        trained[name] = estimator
        results[name] = evaluate(y_test, estimator.predict(X_test_scaled))

    # Tune tree / SVR models.
    for name in [
        "randomforest",
        "gradientboosting",
        "histgradientboosting",
        "svr",
    ]:
        spec = registry.get(name)
        estimator = tune_model(spec, X_train_scaled, y_train.values)
        trained[name] = estimator
        results[name] = evaluate(y_test, estimator.predict(X_test_scaled))

    # Stacking ensemble over the strongest non-linear base learners.
    stacker_estimators = [
        (name, trained[name])
        for name in ["randomforest", "gradientboosting", "ridge"]
        if name in trained
    ]
    stacker = build_stacker(stacker_estimators)
    stacker.fit(X_train_scaled, y_train.values)
    trained["stacker"] = stacker
    results["stacker"] = evaluate(y_test, stacker.predict(X_test_scaled))

    # Select best by RMSE.
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_estimator = trained[best_name]
    logger.info("Best model selected: %s (RMSE=%.4f)", best_name, results[best_name]["rmse"])

    # Refit on full data and wrap in a single pipeline for inference.
    full_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", best_estimator),
        ]
    )
    full_pipe.fit(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
    )

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    save_pipeline(full_pipe, settings.model_path)
    save_pipeline(preprocessor, settings.preprocessor_path)

    metrics_payload = {
        "best_model": best_name,
        "results": results,
        "test_shape": {
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
        },
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=float)
    )

    return full_pipe, metrics_payload, best_name
