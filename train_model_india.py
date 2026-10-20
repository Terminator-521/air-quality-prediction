"""
train_model_india.py
--------------------
Trains one Random Forest regressor for numeric AQI. The displayed CPCB
category is always derived from that displayed AQI, preventing contradictory
number/category pairs in the dashboard.

Feature set is deliberately restricted to the 6 pollutants the live
Open-Meteo API can supply for any location on Earth (PM2.5, PM10, CO,
NO2, SO2, O3) -- not the full CPCB column list -- so the exact same
trained model can be applied to live/forecast readings in dashboard.py,
not just the static historical CSV.
"""

import json
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from config import MODELS_DIR, OUTPUTS_DIR
from data_prep_india import aqi_to_bucket
from database import import_project_history, register_model_run, training_frame

# Must match live_data.COLUMN_MAP's output columns exactly.
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]


def make_regressor():
    return RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )



def evaluate_temporal_holdout(df):
    """Evaluate on the final 20% of dates, never shuffled future rows into training."""
    cutoff = df["Date"].quantile(0.8)
    train = df[df["Date"] <= cutoff]
    test = df[df["Date"] > cutoff]
    model = make_regressor()
    model.fit(train[FEATURES], train["AQI"])
    pred = model.predict(test[FEATURES])
    evaluated = test[["City", "is_crop_burning_season", "AQI"]].copy()
    evaluated["predicted_aqi"] = pred
    evaluated["aqi_band"] = evaluated["AQI"].apply(aqi_to_bucket)

    def metrics(frame):
        actual = frame["AQI"]
        predicted = frame["predicted_aqi"]
        return {
            "rows": int(len(frame)),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "r2": round(float(r2_score(actual, predicted)), 3) if len(frame) > 1 else None,
            "cpcb_category_accuracy": round(float(accuracy_score(
                actual.apply(aqi_to_bucket), [aqi_to_bucket(value) for value in predicted]
            )), 3),
        }

    by_city = {city: metrics(group) for city, group in evaluated.groupby("City")}
    by_season = {
        ("crop_burning_season" if season else "rest_of_year"): metrics(group)
        for season, group in evaluated.groupby("is_crop_burning_season")
    }
    by_aqi_band = {band: metrics(group) for band, group in evaluated.groupby("aqi_band")}
    overall = metrics(evaluated)
    return {
        "evaluation": "chronological holdout: final 20% of dates",
        "train_end_date": str(train["Date"].max().date()),
        "test_start_date": str(test["Date"].min().date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **overall,
        "breakdown_by_city": by_city,
        "breakdown_by_season": by_season,
        "breakdown_by_actual_aqi_band": by_aqi_band,
    }


def main():
    import_project_history()
    df = training_frame().sort_values("Date").reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} verified rows are available; need at least 100 to train.")
    report = evaluate_temporal_holdout(df)
    print("=== India AQI chronological-holdout evaluation ===")
    for key in ("evaluation", "train_end_date", "test_start_date", "train_rows", "test_rows", "mae", "r2", "cpcb_category_accuracy"):
        value = report[key]
        print(f"{key}: {value}")

    # Production model uses all available 2015-2020 CPCB training rows.
    reg = make_regressor()
    reg.fit(df[FEATURES], df["AQI"])
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned_model_path = MODELS_DIR / f"india_aqi_regressor_{version}.joblib"
    joblib.dump(reg, versioned_model_path, compress=3)
    joblib.dump(reg, MODELS_DIR / "india_aqi_regressor.joblib", compress=3)
    joblib.dump(FEATURES, MODELS_DIR / "india_features.joblib")
    (OUTPUTS_DIR / "model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    register_model_run(
        len(df), {"verified_training_rows": len(df), "training_eligible_places": int(df["City"].nunique())},
        report, versioned_model_path, version_label=version, features=FEATURES,
    )
    print("\nSaved regressor/features to models/ and evaluation report to outputs/model_evaluation.json")


if __name__ == "__main__":
    main()
