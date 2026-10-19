import pandas as pd

import forecast_tracker


def test_tracker_records_and_summarises_matured_forecasts(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_tracker, "FORECAST_TRACKER_PATH", tmp_path / "tracker.sqlite")
    location = {"latitude": 28.6139, "longitude": 77.2090}
    forecast = pd.DataFrame({
        "time": pd.to_datetime(["2026-10-20 12:00"]),
        "is_forecast": [True], "predicted_aqi": [180],
    })
    forecast_tracker.record(location, forecast)
    refreshed = forecast.assign(is_forecast=False, predicted_aqi=160)
    forecast_tracker.record(location, refreshed)
    result = forecast_tracker.summary(location)
    assert result["samples"] == 1
    assert result["mae"] == 20.0
