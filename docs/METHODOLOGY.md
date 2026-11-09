# Methodology — Rail ETA v3.0

This document describes the machine learning approach, feature engineering,
model selection, and evaluation strategy used by Rail ETA.

## Problem Statement

Predict **train journey duration** (in hours) from two route-level features:

- `Total_Distance` — total route distance in kilometres.
- `Number_of_Stations` — count of stations/stops on the route.

## Data Source

`data/Dataset1.csv` contains station-level records for multiple trains:

| Column | Description |
|---|---|
| `SN` | Station sequence number within a train. |
| `Train_No` | Unique train identifier. |
| `Station_Name` | Name of the station. |
| `Route_Number` | Route number. |
| `Arrival_time` | Arrival time at the station (`HH:MM:SS`). |
| `Departure_Time` | Departure time from the station (`HH:MM:SS`). |
| `Distance` | Cumulative distance from the origin station (km). |

## Preprocessing

### 1. Schema Validation

The dataset is validated against a required column list. Missing columns raise
a `SchemaValidationError`.

### 2. Time Parsing

`Arrival_time` and `Departure_Time` are parsed from `HH:MM:SS` strings into
`pandas.Timestamp` objects using the configured format (`%H:%M:%S`).

### 3. Midnight Crossing Correction

Trains that cross midnight produce timestamps that appear to go backward
(e.g., departure at `23:00`, arrival at `01:00` the next day). Rail ETA
rolls timestamps forward by 24 hours whenever a timestamp is less than the
previous one, ensuring monotonically non-decreasing journey time.

### 4. Journey Duration Computation

For each train, the journey duration is computed from:

> **First station departure → Final station arrival**

Intermediate stop dwell times are included naturally. The result is converted
from `Timedelta` to hours:

```
Journey_Duration_Hours = Total_Journey_Duration.total_seconds() / 3600
```

### 5. Train-Level Aggregation

Station-level records are aggregated to one row per train:

| Feature | Derivation |
|---|---|
| `Start_Station` | First station by sequence number. |
| `End_Station` | Last station by sequence number. |
| `Total_Distance` | Maximum `Distance` value for the train. |
| `Number_of_Stations` | Count of stations for the train. |
| `Journey_Duration_Hours` | Computed journey duration in hours. |

Rows with missing values or non-positive distance/duration are dropped.

## Feature Engineering

Two numeric features are used for modelling:

1. `Total_Distance`
2. `Number_of_Stations`

Features are standardised (`StandardScaler`) before being passed to linear
models and SVR. Tree-based models receive the same scaled features (scaling
does not harm tree splits and keeps the preprocessing pipeline uniform).

## Modelling Approach

### Model Zoo

Eight regressors are trained and compared:

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Ridge | `alpha` |
| Lasso | `alpha` |
| ElasticNet | `alpha`, `l1_ratio` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| HistGradientBoosting | `max_iter`, `learning_rate`, `max_depth` |
| SVR | `C`, `gamma` |

### Hyperparameter Tuning

Tree-based and regularised models are tuned using `RandomizedSearchCV` with
3-fold cross-validation, optimising for negative root mean squared error
(`neg_root_mean_squared_error`).

### Stacking Ensemble

A `StackingRegressor` combines the three strongest non-linear base learners
(Random Forest, Gradient Boosting, Ridge) with a `LinearRegression` meta-
learner. The stacker is refit on the full training data.

### Model Selection

The best model is selected by **lowest RMSE** on a held-out test split
(20% of data, `random_state=42`).

### Final Artefact

A single `scikit-learn` `Pipeline` is saved:

```
Pipeline(steps=[
    ("preprocessor", ColumnTransformer(...)),
    ("regressor", <best_estimator>)
])
```

This pipeline is persisted with `joblib` to `models/rail_eta_best_model.joblib`
and contains both preprocessing and the trained regressor.

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | `mean(|y_true - y_pred|)` | Average absolute error in hours. Lower is better. |
| MSE | `mean((y_true - y_pred)²)` | Mean squared error. Penalises large errors. |
| RMSE | `sqrt(MSE)` | Same units as target (hours). Lower is better. |
| R² | `1 - SS_res / SS_tot` | Proportion of variance explained. Closer to 1 is better. |

Metrics are computed per model on the test split and saved to
`results/metrics.json`.

## Visualisations

The following charts are generated after training:

| Figure | Description |
|---|---|
| `duration_vs_distance.png` | Scatter: Journey Duration vs Total Distance. |
| `duration_vs_stations.png` | Scatter: Journey Duration vs Number of Stations. |
| `top_routes.png` | Bar chart: Top 10 longest train routes. |
| `actual_vs_predicted_<model>.png` | Scatter: Actual vs Predicted duration. |
| `residuals_<model>.png` | Scatter: Residuals vs Predicted duration. |

## Reproducibility

- `random_state=42` is used for train/test split and all estimators.
- `n_jobs=-1` enables parallel processing where supported.
- The Dockerfile pins the Python version and installs dependencies from
  `requirements.txt` for environment consistency.

## Limitations

- Only two features are used; adding train type, category, or historical
  delay data could improve accuracy.
- The dataset is assumed to be complete per train; trains with missing
  intermediate records may produce skewed durations.
- Midnight crossing assumes a single 24-hour rollforward; journeys spanning
  more than one day are not explicitly handled beyond repeated rollforward.
