import database


def test_mysql_adapter_translates_placeholders_ignore_and_upserts():
    translated = database._MySQLCursor._sql(
        "INSERT OR IGNORE INTO observations (place) VALUES (?)"
    )
    assert translated == "INSERT IGNORE INTO observations (place) VALUES (%s)"

    upsert = database._MySQLCursor._sql(
        "INSERT INTO refreshed_estimates (estimated_aqi) VALUES (?) "
        "ON CONFLICT(latitude, longitude, target_time) DO UPDATE SET "
        "estimated_aqi=excluded.estimated_aqi"
    )
    assert "ON DUPLICATE KEY UPDATE" in upsert
    assert "estimated_aqi=VALUES(estimated_aqi)" in upsert


def test_mysql_database_url_detection(monkeypatch):
    monkeypatch.setattr(database, "DATABASE_URL", "mysql://user:password@db.example:3306/aqi")
    assert database._is_mysql()
    assert not database._is_postgres()
