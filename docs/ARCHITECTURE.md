# Architecture — Rail ETA v3.0

## Overview

Rail ETA predicts train journey duration using route-level features
(total distance and station count). The system is designed for
maintainability, testability, and deployability, targeting a senior
engineering standard.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config` | Centralised, env-overridable Pydantic settings |
| `data.loader` | CSV ingestion + schema validation |
| `preprocessing.time` | Time parsing, midnight-crossing correction, route aggregation |
| `features.builder` | Public feature-engineering entry point |
| `models.base` | Model registry, specs, hyperparameter tuning, stacking, joblib IO |
| `training.pipeline` | Split, train, evaluate, select best, persist pipeline |
| `training.metrics` | MAE/MSE/RMSE/R² computation |
| `evaluation.visuals` | Diagnostic charts |
| `inference.predictor` | Single / batch prediction, duration formatting |
| `api.main` | FastAPI REST service |
| `cli` | Console entry point (train / predict / predict-batch) |
| `app.py` | Streamlit dashboard |

## Data flow

```
raw CSV
  │
  ▼
load_data() ──▶ validate_schema()
  │
  ▼
parse_times() ──▶ sort_records()
  │
  ▼
build_route_features()  (per-train aggregation)
  │
  ▼
train / test split ──▶ StandardScaler
  │
  ▼
8 regressors + RandomizedSearchCV
  │
  ▼
StackingRegressor (RF + GB + Ridge → Linear)
  │
  ▼
Select best by RMSE ──▶ joblib Pipeline (preprocessor + regressor)
```

## Midnight crossing

When a train crosses midnight, arrival/departure timestamps are rolled
forward by one day until they are monotonically non-decreasing within a
single journey. This prevents negative durations and correctly accounts
for overnight travel.

## Artifact contract

- `models/rail_eta_best_model.joblib` — full inference pipeline
- `models/preprocessor.joblib` — fitted `ColumnTransformer`
- `results/metrics.json` — per-model evaluation + best-model name
- `results/features.csv` — train-level feature dataset
- `results/figures/*.png` — EDA and diagnostic charts

## Extensibility

Add a new model by registering a `ModelSpec` in `ModelRegistry._register_defaults`
or by calling `registry.register(ModelSpec(...))` at runtime. The trainer
automatically picks it up.

## Security & ops

- No secrets are logged or hard-coded.
- Configuration is loaded from `.env` / environment variables.
- Dockerfile builds a lean image; `docker-compose` runs API + dashboard.
- GitHub Actions runs lint, type-check, and tests on Python 3.10/3.11.
