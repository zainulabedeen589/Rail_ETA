"""Feature engineering utilities.

Exposes a thin wrapper around the preprocessing module so the rest of the
application depends on a single, stable feature-builder interface.
"""

from __future__ import annotations

import pandas as pd

from config import settings
from logger import get_logger
from time_utils import build_route_features

logger = get_logger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Public entry point for feature construction.

    Args:
        df: Raw dataframe after schema validation.

    Returns:
        Train-level feature dataframe.
    """
    logger.info("Starting feature engineering")
    features = build_route_features(df)
    logger.info("Feature engineering complete: %d rows", len(features))
    return features
