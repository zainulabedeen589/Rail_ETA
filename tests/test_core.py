"""Unit tests for Rail ETA core modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from config import settings
from data_loader import validate_schema
from time_utils import build_route_features, parse_times, sort_records
from metrics import evaluate


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SN": [1, 2, 3, 1, 2],
            "Train_No": [101, 101, 101, 202, 202],
            "Station_Name": ["A", "B", "C", "X", "Y"],
            "Route_Number": [1, 1, 1, 1, 1],
            "Arrival_time": ["00:00:00", "10:00:00", "20:00:00", "00:00:00", "13:00:00"],
            "Departure_Time": ["10:00:00", "10:30:00", "20:30:00", "12:00:00", "13:30:00"],
            "Distance": [0, 50, 100, 0, 30],
        }
    )


def test_validate_schema_passes(sample_df):
    assert validate_schema(sample_df) is sample_df


def test_validate_schema_fails():
    bad = pd.DataFrame({"foo": [1]})
    with pytest.raises(Exception):
        validate_schema(bad)


def test_parse_times(sample_df):
    parsed = parse_times(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(parsed["Arrival_time"])


def test_sort_records(sample_df):
    sorted_df = sort_records(sample_df)
    assert list(sorted_df["Train_No"])[:3] == [101, 101, 101]


def test_build_route_features(sample_df):
    parsed = parse_times(sample_df)
    sorted_df = sort_records(parsed)
    features = build_route_features(sorted_df)
    assert "Journey_Duration_Hours" in features.columns
    assert len(features) == 2


def test_evaluate_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = evaluate(y_true, y_pred)
    assert set(metrics) == {"mae", "mse", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
