import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/multiple_linear_regression_model.pkl")


# ============================================================
# PROJECT TITLE
# ============================================================

print("=" * 45)
print("     TRAIN JOURNEY DURATION PREDICTOR")
print("=" * 45)


# ============================================================
# USER INPUT
# ============================================================

distance = float(input("Enter Total Distance (km): "))
stations = int(input("Enter Number of Stations: "))


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[distance, stations]],
    columns=["Total_Distance", "Number_of_Stations"]
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\nPredicted Journey Duration:")
print(f"{prediction:.2f} hours")


# ============================================================
# CONVERT HOURS INTO DAYS + HOURS
# ============================================================

days = int(prediction // 24)
remaining_hours = prediction % 24


if days > 0:
    print(
        f"Approximately: {days} day(s) "
        f"{remaining_hours:.2f} hour(s)"
    )
else:
    print(
        f"Approximately: {remaining_hours:.2f} hour(s)"
    )


print("\nPrediction completed successfully!")