"""Forecast-revision tracking using the configured shared project database.

Comparisons are between an earlier Open-Meteo model forecast and a later
Open-Meteo estimate, not between a forecast and an official monitor.
"""

from datetime import datetime, timezone

import pandas as pd

import database


def record(location, frame):
    """Store forecast rows and later refreshed rows for one location."""
    database.initialize()
    lat, lon = round(float(location["latitude"]), 3), round(float(location["longitude"]), 3)
    place = ", ".join(part for part in (location.get("name"), location.get("admin1"), location.get("country")) if part)
    now = datetime.now(timezone.utc).isoformat()
    actual = frame[~frame["is_forecast"]]
    reference_time = actual["time"].max() if len(actual) else frame["time"].min()
    forecasts = frame[frame["is_forecast"]]
    refreshed = frame[~frame["is_forecast"]]
    with database.session() as db:
        db.executemany(
            "INSERT OR IGNORE INTO forecasts (latitude, longitude, target_time, issued_at, lead_hours, predicted_aqi, place) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(lat, lon, row.time.isoformat(), now, (row.time - reference_time).total_seconds() / 3600, float(row.predicted_aqi), place) for row in forecasts.itertuples()],
        )
        for row in refreshed.itertuples():
            db.execute("""
                INSERT INTO refreshed_estimates (latitude, longitude, target_time, estimated_aqi, refreshed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(latitude, longitude, target_time) DO UPDATE SET
                    estimated_aqi=excluded.estimated_aqi, refreshed_at=excluded.refreshed_at
            """, (lat, lon, row.time.isoformat(), float(row.predicted_aqi), now))


def summary(location):
    """Return comparison metrics where an older saved forecast has matured."""
    lat, lon = round(float(location["latitude"]), 3), round(float(location["longitude"]), 3)
    database.initialize()
    with database.session() as db:
        rows = db.execute("""
            SELECT f.predicted_aqi, r.estimated_aqi, f.lead_hours
            FROM forecasts f JOIN refreshed_estimates r
              ON f.latitude = r.latitude AND f.longitude = r.longitude AND f.target_time = r.target_time
            WHERE f.latitude = ? AND f.longitude = ? AND f.issued_at < r.refreshed_at AND f.lead_hours IS NOT NULL
        """, (lat, lon)).fetchall()
    if not rows:
        return {"samples": 0}
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["absolute_error"] = (frame["predicted_aqi"] - frame["estimated_aqi"]).abs()
    return {
        "samples": int(len(frame)),
        "mae": round(float(frame["absolute_error"].mean()), 1),
        "by_horizon": [{"horizon": horizon, "mae": round(float(group.absolute_error.mean()), 1), "samples": int(len(group))}
                       for horizon, group in _horizon_groups(frame).items()],
    }


def _horizon_groups(frame):
    labels = pd.cut(frame["lead_hours"], bins=[-float("inf"), 24, 48, float("inf")],
                    labels=["0–24h", "24–48h", "48h+"] , right=False)
    return {label: frame.loc[labels == label] for label in labels.dropna().unique()}


def accuracy_by_location():
    """Matured model-to-model MAE grouped by place and forecast lead time."""
    database.initialize()
    with database.session() as db:
        rows = db.execute("""
            SELECT place, latitude, longitude, lead_time AS "Lead time", COUNT(*) AS "Matured points",
                   ROUND(AVG(absolute_error), 1) AS "MAE (AQI)"
            FROM (
                SELECT f.place, f.latitude, f.longitude,
                       CASE WHEN f.lead_hours < 24 THEN '0–24h'
                            WHEN f.lead_hours < 48 THEN '24–48h' ELSE '48h+' END AS lead_time,
                       ABS(f.predicted_aqi - r.estimated_aqi) AS absolute_error
                FROM forecasts f JOIN refreshed_estimates r
                  ON f.latitude = r.latitude AND f.longitude = r.longitude AND f.target_time = r.target_time
                WHERE f.issued_at < r.refreshed_at AND f.lead_hours IS NOT NULL
            ) comparisons
            GROUP BY place, latitude, longitude, lead_time
            ORDER BY place, CASE lead_time WHEN '0–24h' THEN 0 WHEN '24–48h' THEN 1 ELSE 2 END
        """).fetchall()
    columns = ["Place", "Lead time", "Matured points", "MAE (AQI)"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["Place"] = frame.apply(
        lambda row: row["place"] or f'{row["latitude"]:.3f}, {row["longitude"]:.3f}', axis=1
    )
    return frame[columns].sort_values(["Place", "Lead time"]).reset_index(drop=True)
