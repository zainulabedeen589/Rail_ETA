"""Data loading and schema validation.

Responsible for reading the raw CSV, validating expected columns, and
returning a clean :class:`pandas.DataFrame` ready for preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from config import settings
from exceptions import DataLoadError, SchemaValidationError
from logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS: List[str] = [
    "SN",
    "Train_No",
    "Station_Name",
    "Route_Number",
    "Arrival_time",
    "Departure_Time",
    "Distance",
]


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the train route dataset from CSV.

    Args:
        path: Optional override for the CSV location. Defaults to
            :attr:`settings.data_path`.

    Returns:
        A :class:`pandas.DataFrame` with the raw route records.

    Raises:
        DataLoadError: If the file cannot be read.
    """
    source = Path(path) if path else settings.data_path
    logger.info("Loading dataset from %s", source)

    try:
        df = pd.read_csv(source)
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"Dataset not found at {source}", code="DATA_LOAD_FILE_NOT_FOUND"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(
            f"Failed to read dataset: {exc}", code="DATA_LOAD_READ_ERROR"
        ) from exc

    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the dataframe contains all expected columns.

    Args:
        df: Raw dataframe loaded from CSV.

    Returns:
        The same dataframe if validation passes.

    Raises:
        SchemaValidationError: If required columns are missing.
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns: {missing}",
            code="SCHEMA_MISSING_COLUMNS",
        )
    logger.info("Schema validation passed (%d columns)", len(df.columns))
    return df
