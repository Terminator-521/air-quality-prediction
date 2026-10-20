import sqlite3

import pandas as pd

import database
import forecast_tracker


def test_sqlite_migration_is_idempotent(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite"
    destination = tmp_path / "destination.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE data_imports (source TEXT PRIMARY KEY, imported_at TEXT NOT NULL, row_count INTEGER NOT NULL)")
        db.execute("INSERT INTO data_imports VALUES ('historic', '2026-10-20T00:00:00+00:00', 5)")

    monkeypatch.setattr(database, "AIR_QUALITY_DB_PATH", destination)
    assert database.migrate_sqlite_database(source) == 1
    assert database.migrate_sqlite_database(source) == 0
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT row_count FROM data_imports WHERE source='historic'").fetchone()[0] == 5


def test_alert_subscription_crud_and_daily_deduplication(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "AIR_QUALITY_DB_PATH", tmp_path / "alerts.sqlite")
    location = {"name": "Pune", "country": "India", "latitude": 18.5204, "longitude": 73.8567}
    first = database.save_alert_subscription("person@example.com", location, 150)
    second = database.save_alert_subscription("PERSON@example.com", location, 200)
    assert first == second
    subscriptions = database.list_alert_subscriptions("person@example.com")
    assert len(subscriptions) == 1
    assert subscriptions[0]["threshold"] == 200
    assert database.claim_alert_notification(first, "2026-10-20", 200)
    assert not database.claim_alert_notification(first, "2026-10-20", 200)
    database.release_alert_notification(first, "2026-10-20")
    assert database.claim_alert_notification(first, "2026-10-20", 200)
    assert database.remove_alert_subscription(first, "person@example.com")
    assert database.list_alert_subscriptions("person@example.com") == []


def test_forecast_accuracy_is_grouped_by_place_and_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "AIR_QUALITY_DB_PATH", tmp_path / "accuracy.sqlite")
    location = {"name": "Pune", "admin1": "Maharashtra", "country": "India",
                "latitude": 18.5204, "longitude": 73.8567}
    base = pd.Timestamp("2026-10-20 12:00")
    frame = pd.DataFrame({
        "time": [base + pd.Timedelta(hours=value) for value in (1, 30, 55)],
        "is_forecast": [True, True, True], "predicted_aqi": [100.0, 150.0, 200.0],
    })
    forecast_tracker.record(location, frame)
    refreshed = frame.assign(is_forecast=False, predicted_aqi=[110.0, 130.0, 220.0])
    forecast_tracker.record(location, refreshed)
    report = forecast_tracker.accuracy_by_location()
    assert list(report["Place"]) == ["Pune, Maharashtra, India"] * 3
    assert set(report["Lead time"]) == {"0–24h", "24–48h", "48h+"}
    assert list(report["MAE (AQI)"]) == [10.0, 20.0, 20.0]


def test_official_monitor_snapshots_produce_a_validation_report(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "AIR_QUALITY_DB_PATH", tmp_path / "monitor.sqlite")
    location = {"name": "Pune", "latitude": 18.5204, "longitude": 73.8567}
    reading = {"station_name": "Pune Central", "state": "Maharashtra",
               "observed_at": "2026-10-20T10:00:00+05:30", "aqi": 120, "distance_km": 2.4}
    database.save_official_reading(location, reading, predicted_aqi=135)
    report = database.official_validation_summary()
    assert report.loc[0, "Monitor station"] == "Pune Central"
    assert report.loc[0, "Paired readings"] == 1
    assert report.loc[0, "MAE (AQI)"] == 15
