"""Local forecast-revision tracking for the dashboard.

This intentionally compares a saved Open-Meteo forecast with a later refreshed
Open-Meteo estimate. It is useful for checking forecast stability, but it is
not a substitute for validation against an official ground monitor.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from config import FORECAST_TRACKER_PATH


def _connection():
    FORECAST_TRACKER_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(FORECAST_TRACKER_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            latitude REAL NOT NULL, longitude REAL NOT NULL, target_time TEXT NOT NULL,
            issued_at TEXT NOT NULL, lead_hours REAL, predicted_aqi REAL NOT NULL,
            PRIMARY KEY (latitude, longitude, target_time, issued_at)
        )
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(forecasts)")}
    if "lead_hours" not in columns:
        connection.execute("ALTER TABLE forecasts ADD COLUMN lead_hours REAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS refreshed_estimates (
            latitude REAL NOT NULL, longitude REAL NOT NULL, target_time TEXT NOT NULL,
            estimated_aqi REAL NOT NULL, refreshed_at TEXT NOT NULL,
            PRIMARY KEY (latitude, longitude, target_time)
        )
    """)
    return connection


def record(location, frame):
    """Store forecast rows and later refreshed rows for one location."""
    lat, lon = round(float(location["latitude"]), 3), round(float(location["longitude"]), 3)
    now = datetime.now(timezone.utc).isoformat()
    actual = frame[~frame["is_forecast"]]
    reference_time = actual["time"].max() if len(actual) else frame["time"].min()
    connection = _connection()
    try:
        forecasts = frame[frame["is_forecast"]]
        connection.executemany(
            "INSERT OR IGNORE INTO forecasts (latitude, longitude, target_time, issued_at, lead_hours, predicted_aqi) VALUES (?, ?, ?, ?, ?, ?)",
            [(lat, lon, row.time.isoformat(), now, (row.time - reference_time).total_seconds() / 3600, float(row.predicted_aqi)) for row in forecasts.itertuples()],
        )
        refreshed = frame[~frame["is_forecast"]]
        connection.executemany(
            "INSERT OR REPLACE INTO refreshed_estimates VALUES (?, ?, ?, ?, ?)",
            [(lat, lon, row.time.isoformat(), float(row.predicted_aqi), now) for row in refreshed.itertuples()],
        )
        connection.commit()
    finally:
        connection.close()


def summary(location):
    """Return comparison metrics where an older saved forecast has matured."""
    lat, lon = round(float(location["latitude"]), 3), round(float(location["longitude"]), 3)
    connection = _connection()
    try:
        rows = pd.read_sql_query("""
            SELECT f.predicted_aqi, r.estimated_aqi,
                   f.lead_hours
            FROM forecasts f JOIN refreshed_estimates r
              ON f.latitude = r.latitude AND f.longitude = r.longitude AND f.target_time = r.target_time
            WHERE f.latitude = ? AND f.longitude = ? AND f.issued_at < r.refreshed_at AND f.lead_hours IS NOT NULL
        """, connection, params=(lat, lon))
    finally:
        connection.close()
    if rows.empty:
        return {"samples": 0}
    rows["absolute_error"] = (rows["predicted_aqi"] - rows["estimated_aqi"]).abs()
    return {
        "samples": int(len(rows)),
        "mae": round(float(rows["absolute_error"].mean()), 1),
        "by_horizon": [{"horizon": "0-24h" if lead < 24 else "24h+", "mae": round(float(group.absolute_error.mean()), 1), "samples": int(len(group))}
                       for lead, group in rows.groupby(rows.lead_hours < 24)],
    }
