import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# LEVEL 1 - DATASET LOADING
# ============================================================

# Load dataset
df = pd.read_csv("data/Dataset1.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()


# ============================================================
# LEVEL 1 - TASK 1.2
# TRAIN-WISE ROUTE ANALYSIS
# ============================================================

print("\nNumber of unique trains:")
print(df["Train_No"].nunique())

print("\nRecords per train:")
print(df.groupby("Train_No").size().head(10))

print("\nFirst train records:")

print(
    df[df["Train_No"] == 107][
        [
            "SN",
            "Train_No",
            "Station_Name",
            "Route_Number",
            "Arrival_time",
            "Departure_Time",
            "Distance"
        ]
    ].head(10)
)


# ============================================================
# COMPLETE TRAIN-WISE ROUTE TABLE
# ============================================================

train_routes = (
    df.groupby("Train_No")
    .agg(
        Start_Station=("Station_Name", "first"),
        End_Station=("Station_Name", "last"),
        Total_Distance=("Distance", "max"),
        Number_of_Stations=("Station_Name", "count")
    )
    .reset_index()
)

print("\nTrain-wise Route Table:")
print(train_routes.head(10))


# ============================================================
# VERIFY ROUTE ORDER
# ============================================================

for train_no in [290, 477]:

    print(f"\nRecords for Train {train_no}:")

    train_data = df[df["Train_No"] == train_no][
        [
            "SN",
            "Train_No",
            "Station_Name",
            "Route_Number",
            "Arrival_time",
            "Departure_Time",
            "Distance"
        ]
    ]

    print(train_data.to_string(index=False))


# ============================================================
# LEVEL 1 - TASK 1.3
# DISTANCE STATISTICS
# ============================================================

print("\nDistance Statistics:")
print(df["Distance"].describe())

print("\nNumber of Stations / Stops Statistics:")
print(train_routes["Number_of_Stations"].describe())

print("\nMinimum Number of Stations:")
print(train_routes["Number_of_Stations"].min())

print("\nMaximum Number of Stations:")
print(train_routes["Number_of_Stations"].max())

print("\nAverage Number of Stations:")
print(train_routes["Number_of_Stations"].mean())


# ============================================================
# LEVEL 1 - TASK 1.4
# DATA QUALITY CHECK
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nBlank String Values:")
print(
    df.astype(str)
    .apply(lambda x: x.str.strip() == "")
    .sum()
)

print("\nNegative Distance Values:")
print((df["Distance"] < 0).sum())


# ============================================================
# LEVEL 2 - TASK 2.2
# STANDARDIZE ARRIVAL AND DEPARTURE TIME
# ============================================================

df["Arrival_time"] = pd.to_datetime(
    df["Arrival_time"],
    format="%H:%M:%S"
)

df["Departure_Time"] = pd.to_datetime(
    df["Departure_Time"],
    format="%H:%M:%S"
)

print("\nTime columns after conversion:")

print(
    df[
        [
            "Train_No",
            "Station_Name",
            "Arrival_time",
            "Departure_Time"
        ]
    ].head(10)
)

print("\nTime column data types:")

print(
    df[
        [
            "Arrival_time",
            "Departure_Time"
        ]
    ].dtypes
)


# ============================================================
# LEVEL 2 - TASK 2.3
# STOP DURATION
# ============================================================

# Sort records in proper train and station sequence
df = df.sort_values(
    ["Train_No", "SN"]
).copy()

# Calculate stop duration
df["Stop_Duration"] = (
    df["Departure_Time"] -
    df["Arrival_time"]
)

print("\nTrain 107 Stop Duration:")

print(
    df[df["Train_No"] == 107][
        [
            "Train_No",
            "SN",
            "Station_Name",
            "Arrival_time",
            "Departure_Time",
            "Stop_Duration"
        ]
    ]
)


# ============================================================
# LEVEL 2 - TASK 2.3
# JOURNEY DURATION - TEST
# ============================================================

test_df = df.copy()

test_df = test_df.sort_values(
    ["Train_No", "SN"]
).copy()

# Previous station departure
test_df["Previous_Departure"] = (
    test_df
    .groupby("Train_No")["Departure_Time"]
    .shift(1)
)

# Travel duration
test_df["Travel_Duration"] = (
    test_df["Arrival_time"] -
    test_df["Previous_Departure"]
)

print("\nTrain 107 Journey Test:")

print(
    test_df[test_df["Train_No"] == 107][
        [
            "Train_No",
            "SN",
            "Station_Name",
            "Arrival_time",
            "Departure_Time",
            "Previous_Departure",
            "Travel_Duration"
        ]
    ]
)


# ============================================================
# LEVEL 2 - MIDNIGHT CROSSING TEST
# ============================================================

print("\nTrain 290 Journey Test:")

print(
    test_df[test_df["Train_No"] == 290][
        [
            "Train_No",
            "SN",
            "Station_Name",
            "Arrival_time",
            "Departure_Time",
            "Previous_Departure",
            "Travel_Duration"
        ]
    ]
)


# ============================================================
# LEVEL 2 - HANDLE MIDNIGHT CROSSING
# ============================================================

test_df["Travel_Duration"] = (
    test_df["Arrival_time"] -
    test_df["Previous_Departure"]
)

# Add 24 hours where midnight crossing occurs
test_df.loc[
    test_df["Travel_Duration"] < pd.Timedelta(0),
    "Travel_Duration"
] += pd.Timedelta(days=1)

print("\nTrain 290 Corrected Journey Test:")

print(
    test_df[test_df["Train_No"] == 290][
        [
            "Train_No",
            "SN",
            "Station_Name",
            "Previous_Departure",
            "Arrival_time",
            "Travel_Duration"
        ]
    ]
)


# ============================================================
# TOTAL JOURNEY DURATION PER TRAIN - TEST
# ============================================================

journey_summary = (
    test_df
    .groupby("Train_No")["Travel_Duration"]
    .sum()
    .reset_index(
        name="Total_Journey_Duration"
    )
)

print("\nTotal Journey Duration - Sample:")
print(journey_summary.head(10))

print("\nTrain 290 Total Journey Duration:")

print(
    journey_summary[
        journey_summary["Train_No"] == 290
    ]
)


# ============================================================
# FINAL JOURNEY DURATION CALCULATION
# FIRST DEPARTURE -> FINAL ARRIVAL
# ============================================================

def calculate_journey_duration(group):

    group = group.sort_values("SN").copy()

    # First station departure
    first_departure = group["Departure_Time"].iloc[0]

    current_time = first_departure

    # Process intermediate stations
    for i in range(1, len(group) - 1):

        arrival_time = group["Arrival_time"].iloc[i]

        # Handle midnight crossing
        while arrival_time < current_time:
            arrival_time += pd.Timedelta(days=1)

        departure_time = group["Departure_Time"].iloc[i]

        # Departure cannot be before arrival
        while departure_time < arrival_time:
            departure_time += pd.Timedelta(days=1)

        current_time = departure_time

    # Final station uses ARRIVAL time
    final_arrival = group["Arrival_time"].iloc[-1]

    while final_arrival < current_time:
        final_arrival += pd.Timedelta(days=1)

    return final_arrival - first_departure


journey_duration = (
    test_df
    .groupby("Train_No")
    .apply(calculate_journey_duration)
    .reset_index(
        name="Total_Journey_Duration"
    )
)

print("\nFinal Journey Duration - Sample:")
print(journey_duration.head(10))

print("\nTrain 107:")

print(
    journey_duration[
        journey_duration["Train_No"] == 107
    ]
)

print("\nTrain 290:")

print(
    journey_duration[
        journey_duration["Train_No"] == 290
    ]
)


# ============================================================
# LEVEL 2 - FINAL JOURNEY DURATION FEATURE
# ============================================================

df = df.sort_values(
    ["Train_No", "SN"]
).copy()

journey_duration = (
    df
    .groupby("Train_No")
    .apply(calculate_journey_duration)
    .reset_index(
        name="Total_Journey_Duration"
    )
)

# Convert duration into hours
journey_duration["Journey_Duration_Hours"] = (
    journey_duration["Total_Journey_Duration"]
    .dt.total_seconds()
    / 3600
)

print("\nFinal Journey Duration Dataset:")
print(journey_duration.head(10))

print("\nJourney Duration Statistics:")

print(
    journey_duration[
        "Journey_Duration_Hours"
    ].describe()
)


# ============================================================
# TOP 10 LONGEST JOURNEYS
# ============================================================

longest_journeys = (
    journey_duration
    .sort_values(
        "Journey_Duration_Hours",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Longest Journeys:")
print(longest_journeys)


# ============================================================
# LEVEL 2 - FINAL FEATURE DATASET
# ============================================================

train_features = train_routes.merge(
    journey_duration[
        [
            "Train_No",
            "Journey_Duration_Hours"
        ]
    ],
    on="Train_No",
    how="inner"
)

print("\nFinal Train Feature Dataset:")
print(train_features.head(10))

print("\nDataset Shape:")
print(train_features.shape)

print("\nFinal Columns:")
print(train_features.columns.tolist())


# ============================================================
# LEVEL 3 - EDA CHART 1
# JOURNEY DURATION VS TOTAL DISTANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train_features["Total_Distance"],
    train_features["Journey_Duration_Hours"],
    alpha=0.5
)

plt.xlabel("Total Distance")
plt.ylabel("Journey Duration (Hours)")
plt.title("Journey Duration vs Total Distance")

plt.tight_layout()

plt.savefig(
    "results/journey_duration_vs_distance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA CHART 2
# JOURNEY DURATION VS NUMBER OF STATIONS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    train_features["Number_of_Stations"],
    train_features["Journey_Duration_Hours"],
    alpha=0.5
)

plt.xlabel("Number of Stations / Stops")
plt.ylabel("Journey Duration (Hours)")
plt.title("Journey Duration vs Number of Stations")

plt.tight_layout()

plt.savefig(
    "results/journey_duration_vs_stations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA CHART 3
# TOP 10 LONGEST TRAIN ROUTES
# ============================================================

top_routes = (
    train_features
    .sort_values(
        "Total_Distance",
        ascending=False
    )
    .head(10)
    .copy()
)

top_routes["Route"] = (
    top_routes["Start_Station"]
    + " → "
    + top_routes["End_Station"]
)

plt.figure(figsize=(12, 7))

plt.barh(
    top_routes["Route"],
    top_routes["Total_Distance"]
)

plt.xlabel("Total Distance")
plt.ylabel("Train Route")
plt.title("Top 10 Longest Train Routes")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "results/top_10_longest_train_routes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# EDA - CORRELATION ANALYSIS
# ============================================================

correlation_matrix = train_features[
    [
        "Total_Distance",
        "Number_of_Stations",
        "Journey_Duration_Hours"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ============================================================
# LEVEL 4 - LINEAR REGRESSION
# TRAIN-TEST SPLIT
# ============================================================

X = train_features[
    ["Total_Distance"]
]

y = train_features[
    "Journey_Duration_Hours"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# LINEAR REGRESSION - MODEL TRAINING
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print(
    "\nLinear Regression Model "
    "Trained Successfully!"
)

print("\nModel Coefficient:")
print(model.coef_[0])

print("\nModel Intercept:")
print(model.intercept_)


# ============================================================
# LINEAR REGRESSION - PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

print(
    "\nFirst 10 Actual vs Predicted Values:"
)

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison)


# ============================================================
# LINEAR REGRESSION - EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nLinear Regression Evaluation:")
print("-----------------------------")
print(f"MAE  : {mae:.4f} hours")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} hours")
print(f"R²   : {r2:.4f}")


# ============================================================
# LINEAR REGRESSION
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Journey Duration (Hours)"
)

plt.ylabel(
    "Predicted Journey Duration (Hours)"
)

plt.title(
    "Actual vs Predicted Journey Duration - Linear Regression"
)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted_linear_lr.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# MULTIPLE LINEAR REGRESSION
# ============================================================

X = train_features[
    [
        "Total_Distance",
        "Number_of_Stations"
    ]
]

y = train_features[
    "Journey_Duration_Hours"
]


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "\nMultiple Linear Regression:"
)

print(
    "Training Data Shape:",
    X_train.shape
)

print(
    "Testing Data Shape:",
    X_test.shape
)


# ============================================================
# CREATE AND TRAIN MODEL
# ============================================================

multiple_model = LinearRegression()

multiple_model.fit(
    X_train,
    y_train
)

print(
    "\nMultiple Linear Regression Model "
    "Trained Successfully!"
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\nModel Coefficients:")

print(
    "Total_Distance:",
    multiple_model.coef_[0]
)

print(
    "Number_of_Stations:",
    multiple_model.coef_[1]
)

print("\nModel Intercept:")

print(
    multiple_model.intercept_
)


# ============================================================
# MULTIPLE REGRESSION - PREDICTION
# ============================================================

y_pred_multiple = (
    multiple_model.predict(X_test)
)


# ============================================================
# MULTIPLE REGRESSION - EVALUATION
# ============================================================

mae_multiple = mean_absolute_error(
    y_test,
    y_pred_multiple
)

mse_multiple = mean_squared_error(
    y_test,
    y_pred_multiple
)

rmse_multiple = np.sqrt(
    mse_multiple
)

r2_multiple = r2_score(
    y_test,
    y_pred_multiple
)

print(
    "\nMultiple Linear Regression Evaluation:"
)

print(
    "MAE  :",
    round(mae_multiple, 4),
    "hours"
)

print(
    "MSE  :",
    round(mse_multiple, 4)
)

print(
    "RMSE :",
    round(rmse_multiple, 4),
    "hours"
)

print(
    "R²   :",
    round(r2_multiple, 4)
)


# ============================================================
# MULTIPLE REGRESSION
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    y_test,
    y_pred_multiple,
    alpha=0.5
)

min_value = min(
    y_test.min(),
    y_pred_multiple.min()
)

max_value = max(
    y_test.max(),
    y_pred_multiple.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Journey Duration (Hours)"
)

plt.ylabel(
    "Predicted Journey Duration (Hours)"
)

plt.title(
    "Actual vs Predicted Journey Duration - Multiple Linear Regression"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted_multiple_lr.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_test -
    y_pred_multiple
)

print("\nResidual Statistics:")

print(
    "Mean Residual:",
    round(residuals.mean(), 4)
)

print(
    "Minimum Residual:",
    round(residuals.min(), 4)
)

print(
    "Maximum Residual:",
    round(residuals.max(), 4)
)


# ============================================================
# RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    y_pred_multiple,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Journey Duration (Hours)"
)

plt.ylabel(
    "Residuals (Actual - Predicted)"
)

plt.title(
    "Residual Analysis - Multiple Linear Regression"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/residual_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_path = (
    "models/"
    "multiple_linear_regression_model.pkl"
)

joblib.dump(
    multiple_model,
    model_path
)

print(
    "\nFinal model saved successfully!"
)

print(
    "File:",
    model_path
)


# ============================================================
# MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Multiple Linear Regression"
    ],

    "MAE": [
        mae,
        mae_multiple
    ],

    "RMSE": [
        rmse,
        rmse_multiple
    ],

    "R2": [
        r2,
        r2_multiple
    ]
})

print("\nModel Comparison:")
print(model_comparison)


# ============================================================
# BEST MODEL
# ============================================================

best_model = model_comparison.loc[
    model_comparison["R2"].idxmax()
]

print("\nBest Model:")

print(
    best_model["Model"]
)

print(
    f"Best R² Score: "
    f"{best_model['R2']:.4f}"
)

print(
    f"Best MAE: "
    f"{best_model['MAE']:.4f} hours"
)

print(
    f"Best RMSE: "
    f"{best_model['RMSE']:.4f} hours"
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    model_comparison["Model"],
    model_comparison["R2"]
)

plt.title(
    "Model Comparison - R² Score"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R² Score"
)

plt.ylim(
    0.90,
    1.00
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison_r2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE MODEL EVALUATION RESULTS
# ============================================================

with open(
    "results/evaluation_metrics.txt",
    "w"
) as file:

    file.write(
        "TRAIN JOURNEY DURATION "
        "PREDICTION - MODEL EVALUATION\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        "Linear Regression\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        f"MAE  : {mae:.4f} hours\n"
    )

    file.write(
        f"MSE  : {mse:.4f}\n"
    )

    file.write(
        f"RMSE : {rmse:.4f} hours\n"
    )

    file.write(
        f"R²   : {r2:.4f}\n\n"
    )

    file.write(
        "Multiple Linear Regression\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        f"MAE  : {mae_multiple:.4f} hours\n"
    )

    file.write(
        f"MSE  : {mse_multiple:.4f}\n"
    )

    file.write(
        f"RMSE : {rmse_multiple:.4f} hours\n"
    )

    file.write(
        f"R²   : {r2_multiple:.4f}\n\n"
    )

    file.write(
        "Best Model\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        f"Model: {best_model['Model']}\n"
    )

    file.write(
        f"R²: {best_model['R2']:.4f}\n"
    )

    file.write(
        f"MAE: {best_model['MAE']:.4f} hours\n"
    )

    file.write(
        f"RMSE: {best_model['RMSE']:.4f} hours\n"
    )


# ============================================================
# SAVE FINAL FEATURE DATASET
# ============================================================

train_features.to_csv(
    "results/final_train_features.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "PROJECT ANALYSIS COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print("\nResults saved in:")
print("results/")

print("\nModel saved in:")
print("models/multiple_linear_regression_model.pkl")

print("\nGenerated result files:")

print(
    "1. journey_duration_vs_distance.png"
)

print(
    "2. journey_duration_vs_stations.png"
)

print(
    "3. top_10_longest_train_routes.png"
)

print(
    "4. actual_vs_predicted_linear_lr.png"
)

print(
    "5. actual_vs_predicted_multiple_lr.png"
)

print(
    "6. residual_analysis.png"
)

print(
    "7. model_comparison_r2.png"
)

print(
    "8. evaluation_metrics.txt"
)

print(
    "9. final_train_features.csv"
)

print("\nDone!")