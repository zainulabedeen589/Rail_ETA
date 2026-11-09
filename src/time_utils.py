"""Time preprocessing and midnight-crossing handling.

Converts ``HH:MM:SS`` strings into :class:`pandas.Timestamp` objects and
computes journey duration from first departure to final arrival while
correctly rolling timestamps forward when a train crosses midnight.
"""

from __future__ import annotations

import pandas as pd

from config import settings
from exceptions import PreprocessingError
from logger import get_logger

logger = get_logger(__name__)


def parse_times(df: pd.DataFrame) -> pd.DataFrame:
    """Parse arrival and departure columns into datetime objects.

    Args:
        df: Dataframe containing ``Arrival_time`` and ``Departure_Time``.

    Returns:
        Dataframe with parsed time columns.

    Raises:
        PreprocessingError: If time parsing fails.
    """
    df = df.copy()
    try:
        df["Arrival_time"] = pd.to_datetime(
            df["Arrival_time"], format=settings.time_format, errors="coerce"
        )
        df["Departure_Time"] = pd.to_datetime(
            df["Departure_Time"], format=settings.time_format, errors="coerce"
        )
    except Exception as exc:  # noqa: BLE001
        raise PreprocessingError(
            f"Time parsing failed: {exc}", code="TIME_PARSE_ERROR"
        ) from exc

    null_count = df["Arrival_time"].isna().sum() + df["Departure_Time"].isna().sum()
    if null_count:
        logger.warning("%d null timestamps after parsing", null_count)

    return df


def sort_records(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records by train and station sequence number.

    Args:
        df: Dataframe with ``Train_No`` and ``SN`` columns.

    Returns:
        Sorted dataframe copy.
    """
    return df.sort_values(["Train_No", "SN"]).copy()


def compute_journey_duration(group: pd.DataFrame) -> pd.Timedelta:
    """Compute total journey duration for a single train group.

    The duration spans from the first station's departure to the final
    station's arrival. Intermediate stop dwell times are naturally
    included. Midnight crossings are corrected by advancing timestamps.

    Args:
        group: A slice of the dataframe for one ``Train_No``, already
            sorted by ``SN``.

    Returns:
        A :class:`pandas.Timedelta` representing the journey duration.
    """
    group = group.sort_values("SN").copy()
    if len(group) < 2:
        return pd.Timedelta(0)

    current_time = group["Departure_Time"].iloc[0]

    for i in range(1, len(group) - 1):
        arrival = group["Arrival_time"].iloc[i]
        while pd.notna(arrival) and arrival < current_time:
            arrival += pd.Timedelta(days=settings.midnight_delta_days)

        departure = group["Departure_Time"].iloc[i]
        while pd.notna(departure) and departure < arrival:
            departure += pd.Timedelta(days=settings.midnight_delta_days)

        current_time = departure

    final_arrival = group["Arrival_time"].iloc[-1]
    while pd.notna(final_arrival) and final_arrival < current_time:
        final_arrival += pd.Timedelta(days=settings.midnight_delta_days)

    return final_arrival - current_time


def build_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station-level records into train-level features.

    Args:
        df: Preprocessed dataframe with parsed times.

    Returns:
        Dataframe with one row per train containing
        ``Start_Station``, ``End_Station``, ``Total_Distance``,
        ``Number_of_Stations``, ``Journey_Duration_Hours``.
    """
    train_routes = (
        df.groupby("Train_No")
        .agg(
            Start_Station=("Station_Name", "first"),
            End_Station=("Station_Name", "last"),
            Total_Distance=("Distance", "max"),
            Number_of_Stations=("Station_Name", "count"),
        )
        .reset_index()
    )

    journey = (
        df.groupby("Train_No")
        .apply(compute_journey_duration, include_groups=False)
        .reset_index(name="Total_Journey_Duration")
    )
    journey["Journey_Duration_Hours"] = (
        journey["Total_Journey_Duration"].dt.total_seconds() / 3600.0
    )

    features = train_routes.merge(
        journey[["Train_No", "Journey_Duration_Hours"]],
        on="Train_No",
        how="inner",
    )
    features = features.dropna(
        subset=settings.numeric_features + [settings.target]
    )
    features = features[
        (features["Total_Distance"] > 0) & (features["Journey_Duration_Hours"] > 0)
    ]

    logger.info("Built %d train-level feature rows", len(features))
    return features
