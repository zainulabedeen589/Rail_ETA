"""Evaluation and visualisation helpers.

Generates diagnostic charts and persists evaluation artefacts to disk.
Matplotlib is imported lazily so the package remains importable in
environments without a display backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # noqa: BLE001
    HAS_MATPLOTLIB = False

from config import settings
from logger import get_logger

logger = get_logger(__name__)


def _save_figure(fig, name: str) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib unavailable; skipping figure %s", name)
        return None
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figures_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)
    return path


def plot_duration_vs_distance(features: pd.DataFrame) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        features["Total_Distance"],
        features[settings.target],
        alpha=0.6,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("Total Distance (km)")
    ax.set_ylabel("Journey Duration (hours)")
    ax.set_title("Journey Duration vs Total Distance")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, "duration_vs_distance.png")


def plot_duration_vs_stations(features: pd.DataFrame) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        features["Number_of_Stations"],
        features[settings.target],
        alpha=0.6,
        color="orange",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("Number of Stations")
    ax.set_ylabel("Journey Duration (hours)")
    ax.set_title("Journey Duration vs Number of Stations")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, "duration_vs_stations.png")


def plot_top_routes(features: pd.DataFrame, top_n: int = 10) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None
    top = (
        features.sort_values("Total_Distance", ascending=False)
        .head(top_n)
        .copy()
    )
    top["Route"] = top["Start_Station"] + " \u2192 " + top["End_Station"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Route"], top["Total_Distance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Total Distance (km)")
    ax.set_title(f"Top {top_n} Longest Train Routes")
    ax.grid(True, alpha=0.3, axis="x")
    return _save_figure(fig, "top_routes.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="red")
    ax.set_xlabel("Actual Duration (hours)")
    ax.set_ylabel("Predicted Duration (hours)")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, f"actual_vs_predicted_{model_name}.png")


def plot_residuals(y_true, y_pred, model_name: str) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, linestyle="--", color="red")
    ax.set_xlabel("Predicted Duration (hours)")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Analysis - {model_name}")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, f"residuals_{model_name}.png")


def generate_all_visuals(
    features: pd.DataFrame,
    y_true,
    y_pred,
    model_name: str,
) -> None:
    """Generate the standard suite of evaluation visualisations."""
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(settings.features_path, index=False)
    plot_duration_vs_distance(features)
    plot_duration_vs_stations(features)
    plot_top_routes(features)
    plot_actual_vs_predicted(y_true, y_pred, model_name)
    plot_residuals(y_true, y_pred, model_name)
