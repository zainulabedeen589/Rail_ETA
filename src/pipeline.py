"""Top-level pipeline entry point.

Orchestrates the full end-to-end workflow: load, validate, preprocess,
featurise, train, evaluate, and visualise. Intended to be invoked as::

    python -m src.rail_eta.pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from config import settings
from data_loader import load_data, validate_schema
from exceptions import PreprocessingError
from logger import setup_logging, get_logger
from time_utils import parse_times, sort_records
from builder import build_features
from training_pipeline import split_data, train_all
from visuals import generate_all_visuals
from predictor import load_model

logger = get_logger(__name__)


def run_pipeline(data_path: Path | str | None = None) -> Dict[str, Any]:
    """Execute the complete training pipeline.

    Args:
        data_path: Optional override for the source CSV.

    Returns:
        The metrics dictionary produced by training.
    """
    setup_logging()
    logger.info("Rail ETA training pipeline starting")

    df = load_data(data_path)
    validate_schema(df)

    # Basic data quality guards.
    required = ["Train_No", "Station_Name", "Arrival_time", "Departure_Time", "Distance"]
    missing_vals = df[required].isna().sum().sum()
    if missing_vals:
        logger.warning("Dropping %d rows with missing required values", missing_vals)
        df = df.dropna(subset=required)

    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df = df.dropna(subset=["Distance"])
    df = parse_times(df)
    df = sort_records(df)

    features = build_features(df)
    X_train, X_test, y_train, y_test = split_data(features)

    pipeline, metrics, best_name = train_all(X_train, X_test, y_train, y_test)

    model = load_model(settings.model_path)
    y_pred = model.predict(X_test)
    generate_all_visuals(features, y_test, y_pred, best_name)

    logger.info("Pipeline finished. Best model: %s", best_name)
    return metrics


if __name__ == "__main__":
    run_pipeline()
