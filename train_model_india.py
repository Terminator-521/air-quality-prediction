"""
train_model_india.py
---------------------
Trains a Random Forest classifier to predict the AQI_Bucket category
(Good / Satisfactory / Moderate / Poor / Very Poor / Severe) for Indian
cities from pollutant concentrations, plus a regressor for the numeric
AQI value itself.

Feature set is deliberately restricted to the 6 pollutants the live
Open-Meteo API can supply for any location on Earth (PM2.5, PM10, CO,
NO2, SO2, O3) -- not the full CPCB column list -- so the exact same
trained model can be applied to live/forecast readings in dashboard.py,
not just the static historical CSV.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

from data_prep_india import load_and_clean

# Must match live_data.COLUMN_MAP's output columns exactly.
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]


def main():
    df = load_and_clean()
    X = df[FEATURES]

    # --- Classification: AQI_Bucket ---
    y_cls = df["AQI_Bucket"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("=== India: AQI_Bucket classification ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred))
    joblib.dump(clf, "models/india_aqi_classifier.joblib", compress=3)

    # --- Regression: numeric AQI ---
    y_reg = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print("\n=== India: AQI regression ===")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}, R2: {r2_score(y_test, pred):.3f}")
    joblib.dump(reg, "models/india_aqi_regressor.joblib", compress=3)

    joblib.dump(FEATURES, "models/india_features.joblib")
    print("\nSaved models to models/: india_aqi_classifier.joblib, india_aqi_regressor.joblib, india_features.joblib")


if __name__ == "__main__":
    main()
