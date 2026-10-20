"""SQLite/PostgreSQL/MySQL storage for historical, live, forecast, and verified training data."""

import sqlite3
import json
import re
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from config import AIR_QUALITY_DB_PATH, DATABASE_URL

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]


class _HybridRow(dict):
    """Mapping row that also supports SQLite-style numeric indexing."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _PostgresConnection:
    """Small DB-API adapter so existing qmark SQL works with psycopg."""
    def __init__(self, connection):
        self._connection = connection

    @staticmethod
    def _sql(sql):
        ignore = bool(re.search(r"INSERT\s+OR\s+IGNORE\s+INTO", sql, flags=re.I))
        sql = re.sub(r"INSERT\s+OR\s+IGNORE\s+INTO", "INSERT INTO", sql, flags=re.I)
        sql = sql.replace("?", "%s")
        if ignore and "ON CONFLICT" not in sql.upper():
            sql += " ON CONFLICT DO NOTHING"
        return sql

    def execute(self, sql, params=()):
        return self._connection.execute(self._sql(sql), params)

    def executemany(self, sql, rows):
        return self._connection.cursor().executemany(self._sql(sql), rows)

    def cursor(self):
        return self._connection.cursor()

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


class _MySQLCursor:
    """Cursor adapter providing mapping and SQLite-style positional rows."""
    def __init__(self, cursor):
        self._cursor = cursor

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @staticmethod
    def _sql(sql):
        ignored = bool(re.search(r"INSERT\s+OR\s+IGNORE\s+INTO", sql, flags=re.I))
        sql = re.sub(r"INSERT\s+OR\s+IGNORE\s+INTO", "INSERT IGNORE INTO", sql, flags=re.I)
        sql = sql.replace("?", "%s")
        # MySQL's equivalent syntax for these existing SQLite/PostgreSQL upserts.
        sql = re.sub(r"ON CONFLICT\s*\([^)]*\)\s*DO UPDATE SET", "ON DUPLICATE KEY UPDATE", sql, flags=re.I)
        sql = re.sub(r"excluded\.([a-zA-Z_][a-zA-Z0-9_]*)", r"VALUES(\1)", sql, flags=re.I)
        if not ignored:
            sql = re.sub(r"ON CONFLICT\s*\([^)]*\)\s*DO NOTHING", "", sql, flags=re.I)
        return sql

    def execute(self, sql, params=()):
        self._cursor.execute(self._sql(sql), params)
        return self

    def executemany(self, sql, rows):
        self._cursor.executemany(self._sql(sql), rows)
        return self

    def _row(self, row):
        if row is None:
            return None
        columns = [item[0] for item in self._cursor.description]
        return _HybridRow(zip(columns, row))

    def fetchone(self):
        return self._row(self._cursor.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self._cursor.fetchall()]


class _MySQLConnection:
    """Small DB-API adapter so the shared query layer works with mysql.connector."""
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, params=()):
        return _MySQLCursor(self._connection.cursor()).execute(sql, params)

    def executemany(self, sql, rows):
        return _MySQLCursor(self._connection.cursor()).executemany(sql, rows)

    def cursor(self):
        return _MySQLCursor(self._connection.cursor())

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def _is_postgres():
    return DATABASE_URL.startswith(("postgres://", "postgresql://"))


def _is_mysql():
    return DATABASE_URL.startswith(("mysql://", "mysql+mysqlconnector://"))


def connect():
    if DATABASE_URL and not (_is_postgres() or _is_mysql()):
        raise ValueError("DATABASE_URL must use a PostgreSQL or MySQL URL (postgresql:// or mysql://).")
    if _is_mysql():
        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError("MySQL support requires mysql-connector-python. Install requirements.txt.") from exc
        parsed = urlparse(DATABASE_URL.replace("mysql+mysqlconnector://", "mysql://", 1))
        query = parse_qs(parsed.query)
        options = {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": unquote(parsed.path.lstrip("/")),
            "connection_timeout": 15,
            "autocommit": False,
        }
        # Preserve common TLS options accepted by Connector/Python.
        for key in ("ssl_ca", "ssl_cert", "ssl_key", "ssl_verify_cert", "ssl_verify_identity"):
            if key in query:
                value = query[key][-1]
                options[key] = value.lower() in ("1", "true", "yes") if key.startswith("ssl_verify_") else value
        return _MySQLConnection(mysql.connector.connect(**options))
    if _is_postgres():
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg[binary]. Install requirements.txt.") from exc
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return _PostgresConnection(psycopg.connect(url, row_factory=_hybrid_row, connect_timeout=15))
    AIR_QUALITY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(AIR_QUALITY_DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _hybrid_row(cursor, values):
    return _HybridRow((column.name, value) for column, value in zip(cursor.description, values))


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
    id_type = "BIGSERIAL PRIMARY KEY" if _is_postgres() else "BIGINT AUTO_INCREMENT PRIMARY KEY" if _is_mysql() else "INTEGER PRIMARY KEY"
    subscription_id_type = "BIGSERIAL PRIMARY KEY" if _is_postgres() else "BIGINT AUTO_INCREMENT PRIMARY KEY" if _is_mysql() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    text_type = "VARCHAR(255)" if _is_mysql() else "TEXT"
    location_type = "VARCHAR(128)" if _is_mysql() else "TEXT"
    time_type = "VARCHAR(64)" if _is_mysql() else "TEXT"
    source_type = "VARCHAR(128)" if _is_mysql() else "TEXT"
    with session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id {id_type},
                place {location_type} NOT NULL,
                state {location_type},
                latitude REAL,
                longitude REAL,
                observed_at {time_type} NOT NULL,
                captured_at {time_type} NOT NULL,
                source {source_type} NOT NULL,
                record_type {source_type} NOT NULL,
                pm25 REAL, pm10 REAL, no2 REAL, so2 REAL, o3 REAL, co REAL,
                aqi REAL,
                predicted_aqi REAL,
                verified INTEGER NOT NULL DEFAULT 0,
                training_eligible INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                UNIQUE(place, observed_at, source, record_type, captured_at)
            )
        """.format(id_type=id_type, location_type=location_type, time_type=time_type, source_type=source_type))
        _ensure_index(db, "observations_training", "CREATE INDEX observations_training ON observations(training_eligible, verified)")
        _ensure_index(db, "observations_place_time", "CREATE INDEX observations_place_time ON observations(place, observed_at)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS data_imports (
                source {text_type} PRIMARY KEY, imported_at TEXT NOT NULL, row_count INTEGER NOT NULL
            )
        """.format(text_type=text_type))
        db.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id {id_type}, trained_at TEXT NOT NULL, rows_used INTEGER NOT NULL,
                source_summary TEXT NOT NULL, metrics_json TEXT NOT NULL, model_path TEXT NOT NULL,
                version_label TEXT, features_json TEXT
            )
        """.format(id_type=id_type))
        model_columns = _table_columns(db, "model_runs")
        if "version_label" not in model_columns:
            db.execute("ALTER TABLE model_runs ADD COLUMN version_label TEXT")
        if "features_json" not in model_columns:
            db.execute("ALTER TABLE model_runs ADD COLUMN features_json TEXT")
        db.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                latitude REAL NOT NULL, longitude REAL NOT NULL, target_time {time_type} NOT NULL,
                issued_at {time_type} NOT NULL, lead_hours REAL, predicted_aqi REAL NOT NULL,
                place {location_type},
                PRIMARY KEY (latitude, longitude, target_time, issued_at)
            )
        """.format(time_type=time_type, location_type=location_type))
        if "place" not in _table_columns(db, "forecasts"):
            db.execute("ALTER TABLE forecasts ADD COLUMN place TEXT")
        db.execute("""
            CREATE TABLE IF NOT EXISTS refreshed_estimates (
                latitude REAL NOT NULL, longitude REAL NOT NULL, target_time {time_type} NOT NULL,
                estimated_aqi REAL NOT NULL, refreshed_at {time_type} NOT NULL,
                PRIMARY KEY (latitude, longitude, target_time)
            )
        """.format(time_type=time_type))
        db.execute("""
            CREATE TABLE IF NOT EXISTS alert_subscriptions (
                id {subscription_id_type}, email {text_type} NOT NULL, location_name {location_type} NOT NULL,
                latitude REAL NOT NULL, longitude REAL NOT NULL, threshold INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
                UNIQUE(email, latitude, longitude)
            )
        """.format(subscription_id_type=subscription_id_type, text_type=text_type, location_type=location_type))
        db.execute("""
            CREATE TABLE IF NOT EXISTS alert_notifications (
                subscription_id {subscription_id_type}, alert_date {time_type} NOT NULL, threshold INTEGER NOT NULL,
                notified_at TEXT NOT NULL, PRIMARY KEY(subscription_id, alert_date)
            )
        """.format(subscription_id_type="BIGINT" if (_is_postgres() or _is_mysql()) else "INTEGER", time_type=time_type))


def _table_columns(db, table):
    if _is_postgres():
        return {row["column_name"] for row in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=?",
            (table,),
        )}
    if _is_mysql():
        return {row[0] for row in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=?",
            (table,),
        )}
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _ensure_index(db, name, create_sql):
    if _is_mysql():
        exists = db.execute(
            "SELECT 1 FROM information_schema.statistics WHERE table_schema=DATABASE() AND index_name=? LIMIT 1",
            (name,),
        ).fetchone()
        if not exists:
            db.execute(create_sql)
    else:
        db.execute(create_sql.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1))


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
        forecast_columns = _table_columns(db, "forecasts")
        if "lead_hours" not in forecast_columns:
            db.execute("ALTER TABLE forecasts ADD COLUMN lead_hours REAL")
        db.executemany("INSERT OR IGNORE INTO forecasts (latitude,longitude,target_time,issued_at,lead_hours,predicted_aqi) VALUES (?,?,?,?,?,?)",
                       [(r["latitude"], r["longitude"], r["target_time"], r["issued_at"], r["lead_hours"] if "lead_hours" in r.keys() else None, r["predicted_aqi"]) for r in old_forecasts])
        db.executemany("INSERT OR IGNORE INTO refreshed_estimates VALUES (?,?,?,?,?)",
                       [(r["latitude"], r["longitude"], r["target_time"], r["estimated_aqi"], r["refreshed_at"]) for r in old_refreshed])
        db.execute("INSERT OR IGNORE INTO data_imports VALUES (?, ?, ?)",
                   ("legacy_forecast_tracker", datetime.now(timezone.utc).isoformat(), len(old_forecasts) + len(old_refreshed)))
    return len(old_forecasts) + len(old_refreshed)


def migrate_sqlite_database(path):
    """Copy a local SQLite database into the currently configured database once."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {source_path}")
    if not (_is_postgres() or _is_mysql()) and source_path == AIR_QUALITY_DB_PATH.resolve():
        raise ValueError("Source SQLite database is already the configured destination.")
    marker = "sqlite_migration:" + str(source_path)
    initialize()
    with session() as db:
        if db.execute("SELECT 1 FROM data_imports WHERE source=?", (marker,)).fetchone():
            return 0

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    tables = {
        "observations": ("place,state,latitude,longitude,observed_at,captured_at,source,record_type,pm25,pm10,no2,so2,o3,co,aqi,predicted_aqi,verified,training_eligible,metadata_json").split(","),
        "data_imports": "source,imported_at,row_count".split(","),
        "model_runs": "trained_at,rows_used,source_summary,metrics_json,model_path,version_label,features_json".split(","),
        "forecasts": "latitude,longitude,target_time,issued_at,lead_hours,predicted_aqi,place".split(","),
        "refreshed_estimates": "latitude,longitude,target_time,estimated_aqi,refreshed_at".split(","),
        "alert_subscriptions": "id,email,location_name,latitude,longitude,threshold,active,created_at".split(","),
        "alert_notifications": "subscription_id,alert_date,threshold,notified_at".split(","),
    }
    count = 0
    try:
        with session() as target:
            for table, expected_columns in tables.items():
                try:
                    source_columns = {row[1] for row in source.execute(f"PRAGMA table_info({table})")}
                except sqlite3.OperationalError:
                    continue
                columns = [column for column in expected_columns if column in source_columns]
                if not columns:
                    continue
                rows = source.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()
                column_list = ",".join(columns)
                placeholders = ",".join("?" for _ in columns)
                target.executemany(
                    f"INSERT OR IGNORE INTO {table} ({column_list}) VALUES ({placeholders})",
                    [tuple(row[column] for column in columns) for row in rows],
                )
                count += len(rows)
            if _is_postgres():
                target.execute("SELECT setval(pg_get_serial_sequence('alert_subscriptions','id'), COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM alert_subscriptions")
            target.execute("INSERT OR IGNORE INTO data_imports VALUES (?, ?, ?)",
                           (marker, datetime.now(timezone.utc).isoformat(), count))
    finally:
        source.close()
    return count


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


def save_official_reading(location, reading, *, verified=False, predicted_aqi=None):
    """Store a monitor reading and its optional contemporaneous model estimate."""
    initialize()
    place = str(reading["station_name"])
    pollutants = reading.get("pollutants") or {}
    captured = f"official_capture:{reading['observed_at']}"
    metadata = {"requested_place": location.get("name"), "distance_km": reading.get("distance_km")}
    with session() as db:
        db.execute("""
            INSERT OR IGNORE INTO observations
            (place,state,latitude,longitude,observed_at,captured_at,source,record_type,
             pm25,pm10,no2,so2,o3,co,aqi,predicted_aqi,verified,training_eligible,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (place, reading.get("state"), location.get("latitude"), location.get("longitude"),
              str(reading["observed_at"]), captured, "official_monitor", "official_observation",
              pollutants.get("PM2.5"), pollutants.get("PM10"), pollutants.get("NO2"),
              pollutants.get("SO2"), pollutants.get("O3"), pollutants.get("CO"),
              float(reading["aqi"]), float(predicted_aqi) if predicted_aqi is not None else None,
              int(verified), int(verified and all(pollutants.get(k) is not None for k in FEATURES)),
              json.dumps(metadata))
        )


def official_validation_summary():
    """Summarize saved Open-Meteo AQI estimates paired with official readings."""
    initialize()
    with session() as db:
        rows = db.execute("""
            SELECT place AS station, observed_at, aqi AS monitor_aqi,
                   predicted_aqi AS model_aqi, metadata_json
            FROM observations
            WHERE source='official_monitor' AND record_type='official_observation'
              AND aqi IS NOT NULL AND predicted_aqi IS NOT NULL
        """).fetchall()
    columns = ["Monitor station", "Paired readings", "MAE (AQI)"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["absolute_error"] = (frame["monitor_aqi"] - frame["model_aqi"]).abs()
    report = frame.groupby("station").agg(
        **{"Paired readings": ("absolute_error", "size"), "MAE (AQI)": ("absolute_error", "mean")}
    ).reset_index().rename(columns={"station": "Monitor station"})
    report["MAE (AQI)"] = report["MAE (AQI)"].round(1)
    return report[columns].sort_values("Paired readings", ascending=False).reset_index(drop=True)


def save_alert_subscription(email, location, threshold):
    """Create or update a daily AQI email alert for an email/location pair."""
    initialize()
    email = str(email).strip().lower()
    if "@" not in email or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    threshold = int(threshold)
    if not 0 <= threshold <= 500:
        raise ValueError("AQI threshold must be between 0 and 500.")
    lat, lon = round(float(location["latitude"]), 5), round(float(location["longitude"]), 5)
    name = ", ".join(part for part in (location.get("name"), location.get("admin1"), location.get("country")) if part)
    with session() as db:
        db.execute("""
            INSERT INTO alert_subscriptions (email,location_name,latitude,longitude,threshold,active,created_at)
            VALUES (?,?,?,?,?,1,?)
            ON CONFLICT(email,latitude,longitude) DO UPDATE SET
                location_name=excluded.location_name, threshold=excluded.threshold, active=1
        """, (email, name, lat, lon, threshold, datetime.now(timezone.utc).isoformat()))
        row = db.execute("SELECT id FROM alert_subscriptions WHERE email=? AND latitude=? AND longitude=?",
                         (email, lat, lon)).fetchone()
    return row[0]


def list_alert_subscriptions(email=None):
    """List active subscriptions, optionally limited to one email address."""
    initialize()
    with session() as db:
        if email:
            rows = db.execute("SELECT id,email,location_name,latitude,longitude,threshold FROM alert_subscriptions WHERE active=1 AND email=? ORDER BY location_name",
                              (str(email).strip().lower(),)).fetchall()
        else:
            rows = db.execute("SELECT id,email,location_name,latitude,longitude,threshold FROM alert_subscriptions WHERE active=1 ORDER BY location_name").fetchall()
    return [dict(row) for row in rows]


def remove_alert_subscription(subscription_id, email):
    """Remove an alert only when the supplied email matches its subscription."""
    initialize()
    with session() as db:
        cursor = db.execute("DELETE FROM alert_subscriptions WHERE id=? AND email=?",
                            (int(subscription_id), str(email).strip().lower()))
        return cursor.rowcount > 0


def claim_alert_notification(subscription_id, alert_date, threshold):
    """Atomically reserve one email per subscription per UTC day."""
    initialize()
    with session() as db:
        cursor = db.execute("INSERT OR IGNORE INTO alert_notifications (subscription_id,alert_date,threshold,notified_at) VALUES (?,?,?,?)",
                            (int(subscription_id), str(alert_date), int(threshold), datetime.now(timezone.utc).isoformat()))
        return cursor.rowcount > 0


def release_alert_notification(subscription_id, alert_date):
    """Allow a failed daily delivery to retry on the next collector run."""
    with session() as db:
        db.execute("DELETE FROM alert_notifications WHERE subscription_id=? AND alert_date=?",
                   (int(subscription_id), str(alert_date)))


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
        forecast_count = db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
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
    migrator = sub.add_parser("migrate-sqlite", help="Copy a local SQLite database to the configured DATABASE_URL")
    migrator.add_argument("path", help="Path to the source SQLite database")
    args = parser.parse_args()
    if args.command == "init":
        count = import_project_history()
        moved = migrate_forecast_tracker()
        target = "configured hosted database" if (DATABASE_URL and (_is_mysql() or _is_postgres())) else AIR_QUALITY_DB_PATH
        print(f"Database ready at {target}; imported {count} historical rows and migrated {moved} forecast records.")
    elif args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "migrate-sqlite":
        count = migrate_sqlite_database(args.path)
        print(f"Migrated {count} records from {args.path}.")
    else:
        count = import_verified_csv(args.path, args.source)
        print(f"Imported {count} verified training rows from {args.source}.")
