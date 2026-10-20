"""Scheduled Open-Meteo forecast collector; run with ``python collector.py``."""

import json
import sys
import time
from datetime import datetime, timezone

import joblib

import database
from config import COLLECTOR_LOCATIONS, MODELS_DIR
from forecast_tracker import record
from live_data import LiveDataError, add_predictions, fetch_air_quality
from notifications import email_alert
from station_data import OfficialMonitorError, configured as official_monitor_configured, fetch_nearest_monitor


def _locations():
    try:
        locations = json.loads(COLLECTOR_LOCATIONS)
    except json.JSONDecodeError as exc:
        raise ValueError("AQI_COLLECTOR_LOCATIONS must be a JSON array of location objects.") from exc
    if not isinstance(locations, list):
        raise ValueError("AQI_COLLECTOR_LOCATIONS must be a JSON array of location objects.")
    for item in locations:
        if not isinstance(item, dict) or not {"name", "latitude", "longitude"} <= item.keys():
            raise ValueError("Each collector location needs name, latitude, and longitude fields.")
        lat, lon = float(item["latitude"]), float(item["longitude"])
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Coordinates are out of range for {item['name']}.")
        item["latitude"], item["longitude"] = lat, lon
    return locations


def main():
    locations = _locations()
    model = joblib.load(MODELS_DIR / "india_aqi_regressor.joblib")
    features = joblib.load(MODELS_DIR / "india_features.joblib")
    database.import_project_history()
    database.migrate_forecast_tracker()
    subscriptions = database.list_alert_subscriptions()
    # Alert subscriptions are automatically included in collection, even if
    # they were not manually listed in AQI_COLLECTOR_LOCATIONS.
    for subscription in subscriptions:
        if not any(round(float(item["latitude"]), 5) == round(subscription["latitude"], 5)
                   and round(float(item["longitude"]), 5) == round(subscription["longitude"], 5)
                   for item in locations):
            locations.append({"name": subscription["location_name"], "latitude": subscription["latitude"],
                              "longitude": subscription["longitude"]})
    if not locations:
        raise ValueError("Configure AQI_COLLECTOR_LOCATIONS or add a saved email alert location.")

    failures = []
    notification_failures = []
    forecast_peaks = {}
    for index, location in enumerate(locations):
        try:
            frame = add_predictions(fetch_air_quality(location["latitude"], location["longitude"]), model, features)
            database.save_modelled_frame(location, frame)
            record(location, frame)
            future = frame[frame["is_forecast"]]
            if not future.empty:
                key = (round(float(location["latitude"]), 5), round(float(location["longitude"]), 5))
                forecast_peaks[key] = {"peak": float(future["predicted_aqi"].max()),
                                       "through": future["time"].max().strftime("%Y-%m-%d %H:%M")}
            print(f"Collected {len(frame)} hourly rows for {location['name']}.", flush=True)
            if official_monitor_configured():
                try:
                    monitor = fetch_nearest_monitor(location["latitude"], location["longitude"])
                    current_rows = frame[~frame["is_forecast"]]
                    estimate = float(current_rows.iloc[-1]["predicted_aqi"]) if not current_rows.empty else None
                    if monitor:
                        database.save_official_reading(location, monitor, verified=False, predicted_aqi=estimate)
                except OfficialMonitorError as exc:
                    print(f"Official monitor unavailable for {location['name']}: {exc}", file=sys.stderr, flush=True)
        except (LiveDataError, OSError, ValueError) as exc:
            failures.append(f"{location['name']}: {exc}")
            print(f"Collection failed for {location['name']}: {exc}", file=sys.stderr, flush=True)
        if index + 1 < len(locations):
            time.sleep(1)
    today = datetime.now(timezone.utc).date().isoformat()
    for subscription in subscriptions:
        key = (round(subscription["latitude"], 5), round(subscription["longitude"], 5))
        forecast = forecast_peaks.get(key)
        if not forecast or forecast["peak"] < subscription["threshold"]:
            continue
        if not database.claim_alert_notification(subscription["id"], today, subscription["threshold"]):
            continue
        try:
            email_alert(subscription["email"], subscription["location_name"], subscription["threshold"],
                        forecast["peak"], forecast["through"])
            print(f"Sent AQI alert for {subscription['location_name']} to the subscribed address.", flush=True)
        except Exception as exc:
            database.release_alert_notification(subscription["id"], today)
            notification_failures.append(f"{subscription['location_name']}: {exc}")
            print(f"Alert email failed for {subscription['location_name']}: {exc}", file=sys.stderr, flush=True)
    if failures and len(failures) == len(locations):
        raise RuntimeError("All configured locations failed: " + "; ".join(failures))
    if failures:
        print(f"Completed with {len(failures)} location failure(s).", file=sys.stderr)
    if notification_failures:
        raise RuntimeError("One or more alert emails failed: " + "; ".join(notification_failures))


if __name__ == "__main__":
    main()
