"""FastAPI inference service.

Provides a REST endpoint for single-record journey duration predictions.
Run with::

    uvicorn src.rail_eta.api.main:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys

# Ensure src/ is importable when launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from inference import load_model, predict_single, format_duration
from logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Rail ETA API",
    description="Train journey duration prediction service",
    version="3.0.0",
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model(settings.model_path)
    return _model


class PredictionRequest(BaseModel):
    distance: float = Field(..., gt=0, description="Total route distance in kilometres")
    stations: int = Field(..., gt=0, description="Number of stations/stops")


class PredictionResponse(BaseModel):
    predicted_duration_hours: float
    predicted_duration_human: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "rail-eta", "version": "3.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict journey duration for a single route."""
    try:
        model = _get_model()
        hours = predict_single(model, req.distance, req.stations)
        return PredictionResponse(
            predicted_duration_hours=round(hours, 4),
            predicted_duration_human=format_duration(hours),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail=str(exc))
