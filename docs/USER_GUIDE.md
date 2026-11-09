# User Guide — Rail ETA v3.0

Rail ETA predicts train journey duration from route distance and station
count. This guide covers the **Streamlit dashboard** and the **CLI**.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Streamlit Dashboard

Launch the web interface:

```bash
PYTHONPATH=python app.py
```

Open `http://localhost:8501` in your browser.

### Tabs

| Tab | Purpose |
|---|---|
| **🔮 Predict** | Enter distance and stations to get a journey duration prediction. |
| **📊 Model Comparison** | View MAE, RMSE, and R² for every trained model. |
| **🧠 Training** | Trigger a full retraining run from the UI. |
| **📈 Visualisations** | Browse EDA charts and diagnostic plots. |

### Making a Prediction

1. Open the **🔮 Predict** tab.
2. Enter **Total Distance (km)** — e.g., `500.0`.
3. Enter **Number of Stations** — e.g., `10`.
4. Click **Predict Duration**.
5. The result appears as hours and a human-readable string
   (e.g., `"9.14 hour(s)"` or `"1 day(s) 13.06 hour(s)"`).

### Model Comparison

The **📊 Model Comparison** tab reads `results/metrics.json` and displays a
sorted table of all models trained in the last run, ranked by RMSE. The best
model is highlighted.

### Training from UI

Click **🔄 Train / Retrain Model** on the **🧠 Training** tab to run the full
pipeline (data load → preprocess → featurise → train → evaluate → visualise).
This may take several minutes depending on dataset size.

---

## 2. Command-Line Interface

Run commands with `PYTHONPATH=src`:

### Train the model

```bash
PYTHONPATH=src python -m pipeline
```

### Single prediction

```bash
PYTHONPATH=src python -m cli predict --distance 500 --stations 10
```

**Output**

```
1.14 hour(s)
```

### Batch prediction from CSV

Create an input CSV with columns `Total_Distance` and `Number_of_Stations`:

```csv
Total_Distance,Number_of_Stations
500,10
2000,30
750,15
```

Run batch prediction:

```bash
PYTHONPATH=src python -m cli predict-batch --input routes.csv --output predictions.csv
```

**Output file** (`predictions.csv`):

```csv
Total_Distance,Number_of_Stations,Predicted_Duration_Hours
500,10,9.14
2000,30,37.06
750,15,12.50
```

---

## 3. Interpreting Results

- **Predicted Duration (hours)** — decimal value from the ML model.
- **Human-readable** — converted to days + hours for journeys longer than
  24 hours.

### Example Scenarios

| Distance (km) | Stations | Predicted Duration | Human Readable |
|---|---|---|---|
| 500 | 10 | ~9.14 hours | 9.14 hour(s) |
| 2000 | 30 | ~37.06 hours | 1 day(s) 13.06 hour(s) |
| 100 | 5 | ~2.30 hours | 2.30 hour(s) |

---

## 4. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Model not found` | Model not trained yet. | Run `PYTHONPATH=src python -m pipeline` first. |
| `Dataset not found` | Wrong working directory. | Run commands from the project root with `PYTHONPATH=src`. |
| `ModuleNotFoundError` | Missing dependencies. | Run `pip install -r requirements.txt`. |
| Streamlit blank screen | `src/` not on path. | Set `PYTHONPATH=src` before running `streamlit run app.py`. |

---

## 5. Input Constraints

| Parameter | Minimum | Maximum | Notes |
|---|---|---|---|
| Distance (km) | 0.1 | None | Must be positive. |
| Stations | 1 | None | Must be a positive integer. |
