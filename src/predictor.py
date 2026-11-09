"""Batch and single-record inference.

Loads a persisted pipeline and exposes type-safe prediction helpers used by
both the CLI and the FastAPI service.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd

from config import settings
from exceptions import InferenceError, ModelNotTrainedError
from logger import get_logger

logger = get_logger(__name__)


def load_model(path: Path | str | None = None):
    """Load the persisted inference pipeline.

    Args:
        path: Optional override. Defaults to :attr:`settings.model_path`.

    Returns:
        The loaded scikit-learn pipeline.

    Raises:
        ModelNotTrainedError: If the model file does not exist.
    """
    model_path = Path(path) if path else settings.model_path
    if not model_path.exists():
        raise ModelNotTrainedError(
            f"Model not found at {model_path}. Train first.",
            code="MODEL_FILE_MISSING",
        )
    return joblib.load(model_path)


def predict_single(model, distance: float, stations: int) -> float:
    """Predict journey duration for a single route.

    Args:
        model: Fitted pipeline.
        distance: Total route distance in kilometres.
        stations: Number of stations/stops.

    Returns:
        Predicted duration in hours.

    Raises:
        InferenceError: If prediction fails.
    """
    try:
        input_df = pd.DataFrame(
            [[float(distance), int(stations)]],
            columns=settings.numeric_features,
        )
        prediction = float(model.predict(input_df)[0])
        if prediction < 0:
            prediction = 0.0
        return prediction
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            f"Single prediction failed: {exc}", code="PREDICT_SINGLE_FAILED"
        ) from exc


def predict_batch(
    model,
    input_path: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Predict durations for every row in a CSV file.

    Args:
        model: Fitted pipeline.
        input_path: CSV with columns matching ``settings.numeric_features``.
        output_path: Destination CSV. Defaults to ``predictions.csv`` next
            to the input file.

    Returns:
        Path to the written output CSV.

    Raises:
        InferenceError: If the input is missing required columns.
    """
    source = Path(input_path)
    dest = Path(output_path) if output_path else source.with_name("predictions.csv")

    df = pd.read_csv(source)
    missing = [c for c in settings.numeric_features if c not in df.columns]
    if missing:
        raise InferenceError(
            f"Missing columns: {missing}", code="PREDICT_BATCH_MISSING_COLUMNS"
        )

    df["Predicted_Duration_Hours"] = model.predict(df[settings.numeric_features])
    df.to_csv(dest, index=False)
    logger.info("Batch predictions written to %s (%d rows)", dest, len(df))
    return dest


def format_duration(hours: float) -> str:
    """Format a duration in hours into a human-readable string."""
    hours = max(0.0, float(hours))
    days = int(hours // 24)
    remaining = hours % 24
    if days > 0:
        return f"{days} day(s) {remaining:.2f} hour(s)"
    return f"{remaining:.2f} hour(s)"
