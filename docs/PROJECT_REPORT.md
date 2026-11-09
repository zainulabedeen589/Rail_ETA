@"

# Train Journey Duration Prediction

## 1. Project Title

**Train Journey Duration Prediction using Machine Learning**

---

## 2. Project Objective

The objective of this project is to analyze train route data and build a machine learning model to predict total train journey duration.

The prediction is based on:

- Total Distance
- Number of Stations / Stops

The project follows an end-to-end Data Science workflow from data loading and preprocessing to model training, evaluation, model saving, and prediction.

---

## 3. Dataset Description

The project uses the train route dataset:

`data/Dataset1.csv`

The dataset contains train-wise station and route information including:

- Train Number
- Station Name
- Route Number
- Arrival Time
- Departure Time
- Distance

The dataset was analyzed to understand train routes, station sequences, distances, and journey timings.

---

## 4. Data Loading and Inspection

The dataset was loaded using Pandas.

Initial analysis included:

- First five records
- Dataset shape
- Column names
- Dataset information
- Number of unique trains
- Records per train

This helped understand the structure and quality of the dataset.

---

## 5. Train-wise Route Analysis

Train records were grouped by `Train_No`.

For each train, the following information was extracted:

- Starting Station
- Ending Station
- Total Distance
- Number of Stations

A train-wise route table was created for further analysis and feature engineering.

---

## 6. Data Validation

The dataset was checked for:

- Missing values
- Duplicate records
- Blank string values
- Negative distance values
- Data consistency

These checks helped ensure that the data was suitable for further processing.

---

## 7. Time Preprocessing

Arrival and departure time columns were converted into Pandas datetime format.

The following columns were processed:

- `Arrival_time`
- `Departure_Time`

This allowed time differences to be calculated accurately.

---

## 8. Journey Duration Calculation

Journey duration was calculated using the first station departure time and the final station arrival time.

The calculation also handles journeys crossing midnight.

For intermediate stations, arrival and departure times were processed sequentially.

The final journey duration was converted from a time duration into hours for machine learning.

Feature created:

`Journey_Duration_Hours`

---

## 9. Feature Engineering

The final train-level feature dataset contains:

- `Train_No`
- `Start_Station`
- `End_Station`
- `Total_Distance`
- `Number_of_Stations`
- `Journey_Duration_Hours`

The machine learning models use:

### Features

- Total Distance
- Number of Stations

### Target

- Journey Duration in Hours

The final feature dataset was saved as:

`results/final_train_features.csv`

---

## 10. Exploratory Data Analysis

Several visualizations were created to understand relationships in the data.

### 10.1 Journey Duration vs Total Distance

File:

`results/journey_duration_vs_distance.png`

This visualization shows the relationship between total train route distance and journey duration.

### 10.2 Journey Duration vs Number of Stations

File:

`results/journey_duration_vs_stations.png`

This visualization shows the relationship between the number of stations/stops and total journey duration.

### 10.3 Top 10 Longest Train Routes

File:

`results/top_10_longest_train_routes.png`

This chart shows the longest train routes based on total distance.

### 10.4 Correlation Analysis

Correlation between:

- Total Distance
- Number of Stations
- Journey Duration

was calculated to understand relationships between the variables.

---

## 11. Linear Regression

A Linear Regression model was trained using:

`Total_Distance`

as the input feature.

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The model was evaluated using:

- MAE
- MSE
- RMSE
- R² Score

The actual and predicted values were also visualized.

Output:

`results/actual_vs_predicted_linear_lr.png`

---

## 12. Multiple Linear Regression

A Multiple Linear Regression model was developed using two features:

- Total Distance
- Number of Stations

The target variable was:

- Journey Duration in Hours

The dataset was divided into 80% training data and 20% testing data using `random_state=42`.

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 13. Multiple Linear Regression Results

The final Multiple Linear Regression model achieved:

- **MAE:** 0.9788 hours
- **RMSE:** 2.6264 hours
- **R²:** 0.9453

The R² score indicates that the model explains a high proportion of the variation in journey duration within the test data.

---

## 14. Actual vs Predicted Analysis

The actual journey duration was compared with the predicted journey duration.

Output:

`results/actual_vs_predicted_multiple_lr.png`

A diagonal reference line was used to represent perfect predictions.

Predictions closer to this line indicate better agreement between actual and predicted journey durations.

---

## 15. Residual Analysis

Residuals were calculated as:

**Residual = Actual - Predicted**

Residual statistics were calculated, including:

- Mean Residual
- Minimum Residual
- Maximum Residual

The residual plot was created to examine prediction errors.

Output:

`results/residual_analysis.png`

---

## 16. Model Comparison

Two models were compared:

| Model                      |    MAE |   RMSE |    R² |
| -------------------------- | -----: | -----: | -----: |
| Linear Regression          | 1.1251 | 2.7913 | 0.9382 |
| Multiple Linear Regression | 0.9788 | 2.6264 | 0.9453 |

The model comparison visualization is available at:

`results/model_comparison_r2.png`

---

## 17. Best Model

Based on the R² score and error metrics, **Multiple Linear Regression** performed better than the Linear Regression model in this project.

The Multiple Linear Regression model achieved:

**R² = 0.9453**

It also produced lower MAE and RMSE than the Linear Regression model.

Therefore, Multiple Linear Regression was selected as the final model.

---

## 18. Final Model Saving

The trained model was saved using Joblib.

Model file:

`models/multiple_linear_regression_model.pkl`

This allows the trained model to be reused for future predictions without retraining it.

---

## 19. Journey Duration Prediction

A separate prediction script was created:

`src/predict_journey.py`

The user provides:

- Total Distance in kilometres
- Number of Stations

The saved Multiple Linear Regression model then predicts the expected journey duration.

The prediction is displayed in hours and, when applicable, converted into days and remaining hours.

Example:

### Input

Total Distance: 500 km

Number of Stations: 10

### Prediction

Approximately:

**9.14 hours**

Another test:

### Input

Total Distance: 2000 km

Number of Stations: 30

### Prediction

Approximately:

**1 day 13.06 hours**

---

## 20. Project Structure

```text
Rail_ETA/
│
├── data/
│   └── Dataset1.csv
│
├── documentation/
│   └── PROJECT_REPORT.md
│
├── models/
│   └── multiple_linear_regression_model.pkl
│
├── results/
│   ├── actual_vs_predicted_linear_lr.png
│   ├── actual_vs_predicted_multiple_lr.png
│   ├── evaluation_metrics.txt
│   ├── final_train_features.csv
│   ├── journey_duration_vs_distance.png
│   ├── journey_duration_vs_stations.png
│   ├── model_comparison_r2.png
│   ├── residual_analysis.png
│   └── top_10_longest_train_routes.png
│
├── src/
│   ├── predict_journey.py
│   └── train_route_analysis.py
│
├── README.md
└── requirements.txt
```
