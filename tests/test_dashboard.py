"""Smoke tests for the Streamlit dashboard without network access."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import live_data

ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_has_only_the_three_kept_tabs(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(live_data, "geocode", lambda *args, **kwargs: [])
    monkeypatch.setattr(live_data, "fetch_air_quality", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(live_data, "fetch_weather", lambda *args, **kwargs: {})

    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90).run()

    assert not app.exception, app.exception
    assert [tab.label for tab in app.tabs] == [
        "🌍 Live Air Quality & Forecast",
        "🗺️ India Map & Compare",
        "ℹ️ Data Sources & Status",
    ]
    assert not any((box.key or "").startswith("assistant") for box in app.text_input)


def test_live_place_picker_remains_available(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(live_data, "geocode", lambda *args, **kwargs: [])
    now = pd.Timestamp.now().floor("h")
    hourly = pd.DataFrame({
        "time": [now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=1), now + pd.Timedelta(hours=5)],
        "PM2.5": [25.0, 30.0, 45.0], "PM10": [40.0, 50.0, 70.0],
        "CO": [0.4, 0.5, 0.6], "NO2": [10.0, 12.0, 16.0],
        "SO2": [5.0, 6.0, 8.0], "O3": [30.0, 35.0, 40.0],
        "is_forecast": [False, True, True],
    })
    monkeypatch.setattr(live_data, "fetch_air_quality", lambda *args, **kwargs: hourly.copy())
    monkeypatch.setattr(live_data, "fetch_weather", lambda *args, **kwargs: {})

    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90).run()

    picker = next(box for box in app.selectbox if box.key == "live_pick")
    assert "Pune" in picker.options
    picker.set_value("Pune").run()
    assert not app.exception, app.exception
    assert any("AQI average" in table.value.columns for table in app.dataframe)


def test_daily_forecast_summary_pairs_aqi_with_pollutant_estimates():
    hourly = pd.DataFrame({
        "time": pd.to_datetime(["2026-10-19 09:00", "2026-10-19 15:00", "2026-10-20 09:00"]),
        "predicted_aqi": [80, 120, 240],
        "PM2.5": [20.0, 40.0, 80.0],
        "PM10": [35.0, 55.0, 100.0],
        "NO2": [10.0, 20.0, 40.0],
        "SO2": [5.0, 7.0, 12.0],
        "O3": [30.0, 50.0, 70.0],
        "CO": [0.3, 0.5, 0.8],
    })

    daily = live_data.daily_forecast_summary(hourly)

    assert len(daily) == 2
    assert daily.loc[0, "AQI average"] == 100
    assert daily.loc[0, "AQI peak"] == 120
    assert daily.loc[0, "Category"] == "Satisfactory"
    assert daily.loc[0, "PM2.5 (µg/m³)"] == 30
    assert daily.loc[0, "CO (mg/m³)"] == 0.4
    assert daily.loc[1, "Category"] == "Poor"


def test_pollutant_trend_summary_compares_next_day_to_current():
    current = pd.Series({"PM2.5": 20.0, "PM10": 40.0, "NO2": 10.0,
                         "SO2": 5.0, "O3": 30.0, "CO": 0.4})
    forecast = pd.DataFrame({
        "time": pd.to_datetime(["2026-10-20 12:00", "2026-10-20 18:00"]),
        "PM2.5": [30.0, 32.0], "PM10": [32.0, 34.0], "NO2": [10.0, 10.0],
        "SO2": [5.0, 5.0], "O3": [30.0, 30.0], "CO": [0.4, 0.4],
    })
    trends = live_data.pollutant_trend_summary(current, forecast).set_index("Pollutant")
    assert trends.loc["PM2.5", "Trend"] == "Rising"
    assert trends.loc["PM10", "Trend"] == "Falling"
    assert trends.loc["CO", "Unit"] == "mg/m³"
