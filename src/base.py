"""Model base classes and registry.

Defines an abstract model interface, concrete model builders, and a
registry used by the trainer to instantiate and tune regressors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import settings
from exceptions import ModelNotTrainedError, ModelRegistryError
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """Specification for a registered model."""

    name: str
    estimator: Callable[..., BaseEstimator]
    param_dist: Dict[str, Any] = field(default_factory=dict)
    fit_params: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 10


class BaseRegressor(ABC):
    """Abstract wrapper around scikit-learn regressors."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRegressor":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class SklearnRegressor(BaseRegressor):
    """Thin wrapper exposing fit/predict for any scikit-learn estimator."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnRegressor":
        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ModelNotTrainedError(
                "Model has not been fitted yet", code="MODEL_NOT_FITTED"
            )
        return np.asarray(self.estimator.predict(X))


def build_preprocessor() -> ColumnTransformer:
    """Build the standard feature preprocessing pipeline.

    Returns:
        A :class:`~sklearn.compose.ColumnTransformer` that standardises
        the configured numeric features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), settings.numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class ModelRegistry:
    """Registry of available regressors with their search spaces."""

    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            ModelSpec(
                name="linear",
                estimator=LinearRegression,
                param_dist={},
            )
        )
        self.register(
            ModelSpec(
                name="ridge",
                estimator=Ridge,
                param_dist={"alpha": [0.1, 1.0, 10.0, 100.0]},
            )
        )
        self.register(
            ModelSpec(
                name="lasso",
                estimator=Lasso,
                param_dist={"alpha": [0.001, 0.01, 0.1, 1.0]},
                fit_params={"max_iter": 10000},
            )
        )
        self.register(
            ModelSpec(
                name="elasticnet",
                estimator=ElasticNet,
                param_dist={
                    "alpha": [0.001, 0.01, 0.1, 1.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                },
                fit_params={"max_iter": 10000},
            )
        )
        self.register(
            ModelSpec(
                name="randomforest",
                estimator=RandomForestRegressor,
                param_dist={
                    "n_estimators": [100, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            )
        )
        self.register(
            ModelSpec(
                name="gradientboosting",
                estimator=GradientBoostingRegressor,
                param_dist={
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
            )
        )
        self.register(
            ModelSpec(
                name="histgradientboosting",
                estimator=HistGradientBoostingRegressor,
                param_dist={
                    "max_iter": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [None, 5],
                },
            )
        )
        self.register(
            ModelSpec(
                name="svr",
                estimator=SVR,
                param_dist={"C": [1, 10, 100], "gamma": ["scale", "auto"]},
            )
        )

    def register(self, spec: ModelSpec) -> None:
        """Register a new model specification."""
        self._specs[spec.name] = spec
        logger.debug("Registered model: %s", spec.name)

    def get(self, name: str) -> ModelSpec:
        """Retrieve a model specification by name."""
        if name not in self._specs:
            raise ModelRegistryError(
                f"Model '{name}' is not registered", code="MODEL_NOT_REGISTERED"
            )
        return self._specs[name]

    def names(self) -> List[str]:
        """Return all registered model names."""
        return list(self._specs.keys())


def tune_model(spec: ModelSpec, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Run a randomized search for a single model spec.

    Args:
        spec: Model specification.
        X: Scaled training features.
        y: Training target.

    Returns:
        The best estimator found by the search.
    """
    base_estimator = spec.estimator(**spec.fit_params)
    search = RandomizedSearchCV(
        base_estimator,
        param_distributions=spec.param_dist,
        n_iter=min(spec.n_iter, max(1, len(list(spec.param_dist.values())[0])) if spec.param_dist else 1),
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=settings.random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    logger.info(
        "Tuned %s - best RMSE: %.4f", spec.name, -search.best_score_
    )
    return search.best_estimator_


def build_stacker(
    estimators: List[tuple[str, BaseEstimator]],
) -> StackingRegressor:
    """Build a stacking regressor over the given base estimators.

    Args:
        estimators: List of ``(name, estimator)`` tuples.

    Returns:
        A configured :class:`~sklearn.ensemble.StackingRegressor`.
    """
    return StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        n_jobs=-1,
    )


def save_pipeline(pipeline: Pipeline, path: Path | str) -> None:
    """Persist a fitted scikit-learn pipeline to disk via joblib.

    Args:
        pipeline: Fitted pipeline.
        path: Destination file path.
    """
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s", path)


def load_pipeline(path: Path | str) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Args:
        path: Path to the joblib file.

    Returns:
        The loaded :class:`~sklearn.pipeline.Pipeline`.
    """
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s", path)
    return pipeline
