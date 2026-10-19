"""SQLite storage for historical, live, forecast, and verified training data."""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from config import AIR_QUALITY_DB_PATH

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]


def connect():
    AIR_QUALITY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(AIR_QUALITY_DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


@contextmanager
def session():
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                place TEXT NOT NULL,
                state TEXT,
                latitude REAL,
                longitude REAL,
                observed_at TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                source TEXT NOT NULL,
                record_type TEXT NOT NULL,
                pm25 REAL, pm10 REAL, no2 REAL, so2 REAL, o3 REAL, co REAL,
                aqi REAL,
                predicted_aqi REAL,
                verified INTEGER NOT NULL DEFAULT 0,
                training_eligible INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                UNIQUE(place, observed_at, source, record_type, captured_at)
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS observations_training ON observations(training_eligible, verified)")
        db.execute("CREATE INDEX IF NOT EXISTS observations_place_time ON observations(place, observed_at)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS data_imports (
                source TEXT PRIMARY KEY, imported_at TEXT NOT NULL, row_count INTEGER NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY, trained_at TEXT NOT NULL, rows_used INTEGER NOT NULL,
                source_summary TEXT NOT NULL, metrics_json TEXT NOT NULL, model_path TEXT NOT NULL,
                version_label TEXT, features_json TEXT
            )
        """)
        model_columns = {row[1] for row in db.execute("PRAGMA table_info(model_runs)")}
        if "version_label" not in model_columns:
            db.execute("ALTER TABLE model_runs ADD COLUMN version_label TEXT")
        if "features_json" not in model_columns:
            db.execute("ALTER TABLE model_runs ADD COLUMN features_json TEXT")


def import_project_history():
    """Import the two bundled history sources once; safe to call on each app start."""
    from data_prep_india import load_and_clean, load_new_data

    initialize()
    inserted = 0
    with session() as db:
        completed = {row[0] for row in db.execute("SELECT source FROM data_imports")}
    metadata_upgrade = "historical_metadata_v1" not in completed
    sources = []
    if "cpcb_city_day_2015_2020" not in completed or metadata_upgrade:
        sources.append(("cpcb_city_day_2015_2020", load_and_clean(), True))
    if "india_daily_aqi_2022_2025" not in completed or metadata_upgrade:
        sources.append(("india_daily_aqi_2022_2025", load_new_data(), False))
    with session() as db:
        for source, frame, eligible in sources:
            rows = []
            for _, series in frame.iterrows():
                get = series.to_dict()
                city = str(get.get("City", ""))
                date = pd.Timestamp(get["Date"]).isoformat()
                def number(key):
                    value = get.get(key)
                    return None if value is None or pd.isna(value) else float(value)
                metadata = {key: (None if value is None or pd.isna(value) else value)
                            for key, value in get.items()
                            if key not in {"City", "State", "Date", "AQI", *FEATURES}}
                rows.append((
                    city, get.get("State"), None, None, date, "historical_import", source,
                    "historical", number("PM2.5"), number("PM10"), number("NO2"),
                    number("SO2"), number("O3"), number("CO"), number("AQI"), None,
                    1 if eligible else 0, 1 if eligible else 0, json.dumps(metadata, default=str),
                ))
            db.executemany("""
                INSERT INTO observations
                (place,state,latitude,longitude,observed_at,captured_at,source,record_type,
                 pm25,pm10,no2,so2,o3,co,aqi,predicted_aqi,verified,training_eligible,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(place,observed_at,source,record_type,captured_at)
                DO UPDATE SET metadata_json=excluded.metadata_json
            """, rows)
            if source not in completed:
                db.execute("INSERT INTO data_imports VALUES (?, ?, ?)", (source, datetime.now(timezone.utc).isoformat(), len(rows)))
            inserted += len(rows)
        if metadata_upgrade:
            db.execute("INSERT OR IGNORE INTO data_imports VALUES (?, ?, ?)",
                       ("historical_metadata_v1", datetime.now(timezone.utc).isoformat(), inserted))
    return inserted


def migrate_forecast_tracker():
    """Copy records from the earlier standalone tracker DB when it exists."""
    from config import DATA_DIR

    legacy_path = DATA_DIR / "forecast_tracking.sqlite"
    if not legacy_path.exists() or legacy_path.resolve() == AIR_QUALITY_DB_PATH.resolve():
        return 0
    with session() as db:
        if db.execute("SELECT 1 FROM data_imports WHERE source='legacy_forecast_tracker'").fetchone():
            return 0
    legacy = sqlite3.connect(legacy_path)
    legacy.row_factory = sqlite3.Row
    try:
        old_forecasts = legacy.execute("SELECT * FROM forecasts").fetchall()
        old_refreshed = legacy.execute("SELECT * FROM refreshed_estimates").fetchall()
    except sqlite3.Error:
        legacy.close()
        return 0
    legacy.close()
    with session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                latitude REAL NOT NULL, longitude REAL NOT NULL, target_time TEXT NOT NULL,
                issued_at TEXT NOT NULL, lead_hours REAL, predicted_aqi REAL NOT NULL,
                PRIMARY KEY (latitude, longitude, target_time, issued_at)
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS refreshed_estimates (
                latitude REAL NOT NULL, longitude REAL NOT NULL, target_time TEXT NOT NULL,
                estimated_aqi REAL NOT NULL, refreshed_at TEXT NOT NULL,
                PRIMARY KEY (latitude, longitude, target_time)
            )
        """)
        forecast_columns = {row[1] for row in db.execute("PRAGMA table_info(forecasts)")}
        if "lead_hours" not in forecast_columns:
            db.execute("ALTER TABLE forecasts ADD COLUMN lead_hours REAL")
        db.executemany("INSERT OR IGNORE INTO forecasts (latitude,longitude,target_time,issued_at,lead_hours,predicted_aqi) VALUES (?,?,?,?,?,?)",
                       [(r["latitude"], r["longitude"], r["target_time"], r["issued_at"], r["lead_hours"] if "lead_hours" in r.keys() else None, r["predicted_aqi"]) for r in old_forecasts])
        db.executemany("INSERT OR IGNORE INTO refreshed_estimates VALUES (?,?,?,?,?)",
                       [(r["latitude"], r["longitude"], r["target_time"], r["estimated_aqi"], r["refreshed_at"]) for r in old_refreshed])
        db.execute("INSERT OR IGNORE INTO data_imports VALUES (?, ?, ?)",
                   ("legacy_forecast_tracker", datetime.now(timezone.utc).isoformat(), len(old_forecasts) + len(old_refreshed)))
    return len(old_forecasts) + len(old_refreshed)


def save_modelled_frame(location, frame, source="open_meteo_cams"):
    """Store pollutant estimates and AQI predictions. Never training-eligible."""
    initialize()
    place = ", ".join(p for p in (location.get("name"), location.get("admin1"), location.get("country")) if p)
    lat, lon = float(location["latitude"]), float(location["longitude"])
    now = datetime.now(timezone.utc)
    # Use a 15-minute capture bucket to avoid duplicates on Streamlit reruns.
    captured_at = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0).isoformat()
    rows = []
    for _, series in frame.iterrows():
        values = series.to_dict()
        rows.append((place, location.get("admin1"), lat, lon, pd.Timestamp(values["time"]).isoformat(),
                     captured_at, source, "forecast" if values.get("is_forecast") else "live_estimate",
                     *(float(values[col]) if pd.notna(values[col]) else None for col in FEATURES),
                     None, float(values["predicted_aqi"]), 0, 0))
    with session() as db:
        db.executemany("""
            INSERT OR IGNORE INTO observations
            (place,state,latitude,longitude,observed_at,captured_at,source,record_type,
             pm25,pm10,no2,so2,o3,co,aqi,predicted_aqi,verified,training_eligible)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


def save_official_reading(location, reading, *, verified=False):
    """Store a normalized station reading. Training requires explicit verification."""
    initialize()
    place = str(reading["station_name"])
    pollutants = reading.get("pollutants") or {}
    captured = f"official_capture:{reading['observed_at']}"
    with session() as db:
        db.execute("""
            INSERT OR IGNORE INTO observations
            (place,state,latitude,longitude,observed_at,captured_at,source,record_type,
             pm25,pm10,no2,so2,o3,co,aqi,verified,training_eligible)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (place, reading.get("state"), location.get("latitude"), location.get("longitude"),
              str(reading["observed_at"]), captured, "official_monitor", "official_observation",
              pollutants.get("PM2.5"), pollutants.get("PM10"), pollutants.get("NO2"),
              pollutants.get("SO2"), pollutants.get("O3"), pollutants.get("CO"),
              float(reading["aqi"]), int(verified), int(verified and all(pollutants.get(k) is not None for k in FEATURES)))
        )


def training_frame():
    """Return only verified rows with all six required pollutant features and AQI."""
    initialize()
    with session() as db:
        frame = pd.read_sql_query("""
            SELECT place AS City, observed_at AS Date, pm25 AS "PM2.5", pm10 AS PM10,
                   no2 AS NO2, so2 AS SO2, o3 AS O3, co AS CO, aqi AS AQI
            FROM observations
            WHERE verified=1 AND training_eligible=1 AND aqi IS NOT NULL
              AND pm25 IS NOT NULL AND pm10 IS NOT NULL AND no2 IS NOT NULL
              AND so2 IS NOT NULL AND o3 IS NOT NULL AND co IS NOT NULL
            ORDER BY observed_at, place
        """, db)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["month"] = frame["Date"].dt.month
    frame["day"] = frame["Date"].dt.day
    frame["year"] = frame["Date"].dt.year
    frame["is_crop_burning_season"] = (((frame["month"] == 10) & (frame["day"] >= 15)) | (frame["month"] == 11))
    return frame


def status():
    initialize()
    with session() as db:
        total = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        by_type = {row[0]: row[1] for row in db.execute("SELECT record_type, COUNT(*) FROM observations GROUP BY record_type")}
        trainable = db.execute("SELECT COUNT(*) FROM observations WHERE verified=1 AND training_eligible=1").fetchone()[0]
        imports = [dict(row) for row in db.execute("SELECT * FROM data_imports ORDER BY source")]
        forecast_count = db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='forecasts'").fetchone() else 0
        model_runs = [dict(row) for row in db.execute("SELECT id,trained_at,rows_used,version_label,model_path FROM model_runs ORDER BY id DESC LIMIT 5")]
    return {"total_observations": total, "by_type": by_type, "forecast_snapshots": forecast_count,
            "training_eligible": trainable, "imports": imports, "recent_model_runs": model_runs}


def register_model_run(rows_used, sources, metrics, model_path, version_label=None, features=None):
    import json

    initialize()
    with session() as db:
        db.execute("INSERT INTO model_runs (trained_at, rows_used, source_summary, metrics_json, model_path, version_label, features_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (datetime.now(timezone.utc).isoformat(), int(rows_used), json.dumps(sources), json.dumps(metrics), str(model_path), version_label, json.dumps(features or [])))


def import_verified_csv(path, source):
    """Import rows with City, Date, six pollutant columns, and AQI as verified labels."""
    frame = pd.read_csv(path, parse_dates=["Date"])
    required = {"City", "Date", "AQI", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")
    initialize()
    captured = f"verified_import:{source}"
    rows = []
    for _, row in frame.iterrows():
        if row[list(required - {"City", "Date"})].isna().any():
            continue
        rows.append((str(row["City"]), row.get("State"), None, None,
                     pd.Timestamp(row["Date"]).isoformat(), captured, source,
                     "verified_training_observation", *(float(row[name]) for name in FEATURES),
                     float(row["AQI"]), None, 1, 1))
    with session() as db:
        db.executemany("""
            INSERT OR IGNORE INTO observations
            (place,state,latitude,longitude,observed_at,captured_at,source,record_type,
             pm25,pm10,no2,so2,o3,co,aqi,predicted_aqi,verified,training_eligible)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Initialize and inspect the local air-quality database.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the database and import bundled historical data once")
    sub.add_parser("status", help="Show observation counts and imported sources")
    importer = sub.add_parser("import-training-csv", help="Import a verified pollutant/AQI CSV for training")
    importer.add_argument("path")
    importer.add_argument("--source", required=True, help="Human-readable dataset provenance")
    args = parser.parse_args()
    if args.command == "init":
        count = import_project_history()
        moved = migrate_forecast_tracker()
        print(f"Database ready at {AIR_QUALITY_DB_PATH}; imported {count} historical rows and migrated {moved} forecast records.")
    elif args.command == "status":
        print(json.dumps(status(), indent=2))
    else:
        count = import_verified_csv(args.path, args.source)
        print(f"Imported {count} verified training rows from {args.source}.")
