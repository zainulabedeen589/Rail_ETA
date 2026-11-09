"""Application configuration managed through Pydantic settings.

All environment variables are read from ``.env`` or the shell environment.
Values have sensible defaults so the package is runnable out-of-the-box
while remaining overridable for staging/production deployments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised application settings."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_path: Path = Field(default=base_dir / "data" / "Dataset1.csv")
    model_dir: Path = Field(default=base_dir / "models")
    results_dir: Path = Field(default=base_dir / "results")
    model_path: Path = Field(default=base_dir / "models" / "rail_eta_best_model.joblib")
    preprocessor_path: Path = Field(
        default=base_dir / "models" / "preprocessor.joblib"
    )
    metrics_path: Path = Field(default=base_dir / "results" / "metrics.json")
    features_path: Path = Field(default=base_dir / "results" / "features.csv")

    # ------------------------------------------------------------------
    # Modelling
    # ------------------------------------------------------------------
    target: str = "Journey_Duration_Hours"
    numeric_features: List[str] = Field(
        default=["Total_Distance", "Number_of_Stations"]
    )
    test_size: float = Field(default=0.20, ge=0.05, le=0.50)
    random_state: int = Field(default=42)
    n_splits: int = Field(default=5, ge=2, le=10)
    time_format: str = "%H:%M:%S"
    midnight_delta_days: int = Field(default=1, ge=1)

    # ------------------------------------------------------------------
    # API / Server
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    # Derived paths (computed after base fields are set)
    figures_dir: Path = Field(default=base_dir / "results" / "figures")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator(
        "data_path",
        "model_path",
        "preprocessor_path",
        "metrics_path",
        "features_path",
        "figures_dir",
        mode="before",
    )
    @classmethod
    def _resolve_paths(cls, v: Path) -> Path:
        return Path(v).resolve()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings singleton."""
    return Settings()


settings = get_settings()
