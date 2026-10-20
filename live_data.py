"""
live_data.py
------------
Fetches live + forecast air quality data from Open-Meteo's free API.
No API key, no signup, no rate-limit token required for reasonable use.

Two endpoints are used:
  - Geocoding API: turn any place name the user types into (lat, lon)
    https://geocoding-api.open-meteo.com/v1/search
  - Air Quality API: current + hourly forecast pollutant concentrations
    for any coordinate on Earth (backed by the CAMS atmospheric model)
    https://air-quality-api.open-meteo.com/v1/air-quality

Docs: https://open-meteo.com/en/docs/air-quality-api

The pollutant set requested here (PM2.5, PM10, CO, NO2, SO2, O3) is
deliberately the subset that Open-Meteo's *global* CAMS model provides
everywhere -- a few extra variables (like ammonia) only exist in the
Europe-only regional model and would silently come back empty for a
location like Delhi or Mumbai.
"""

from datetime import timedelta

import pandas as pd
import requests

from config import AIR_QUALITY_URL, FORECAST_DAYS, GEOCODE_URL, PAST_DAYS, WEATHER_URL
from data_prep_india import aqi_to_bucket

HOURLY_VARS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone",
]

# Open-Meteo's field names -> the column names our CPCB-trained model
# expects. Every pollutant is µg/m³ in both Open-Meteo and the CPCB
# training data EXCEPT carbon monoxide, which Open-Meteo reports in
# µg/m³ but CPCB (and our model) uses mg/m³ -- handled below.
COLUMN_MAP = {
    "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2", "ozone": "O3", "carbon_monoxide": "CO",
}


class LiveDataError(Exception):
    """Raised when the live API can't be reached or returns nothing usable."""


def geocode(query: str, count: int = 5) -> list:
    """
    Look up a place name the user typed and return matching locations as
    a list of dicts: {name, admin1, country, latitude, longitude}.
    Works for any place on Earth, not just a fixed city list.
    """
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": query, "count": count, "language": "en", "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LiveDataError(f"Couldn't reach the location search service ({e}).") from e

    results = resp.json().get("results") or []
    return [
        {
            "name": r["name"],
            "admin1": r.get("admin1", ""),
            "country": r.get("country", ""),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }
        for r in results
    ]


def fetch_air_quality(lat: float, lon: float, past_days: int = PAST_DAYS, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Hourly pollutant concentrations for a coordinate: `past_days` of
    recent actuals plus `forecast_days` ahead, in one call. Returns a
    DataFrame with a Datetime column, one column per pollutant (renamed
    to match the CPCB training columns), and an `is_forecast` flag.
    """
    try:
        resp = requests.get(
            AIR_QUALITY_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(HOURLY_VARS),
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LiveDataError(f"Couldn't reach the air quality service ({e}).") from e

    payload = resp.json()
    hourly = payload.get("hourly")
    if not hourly:
        raise LiveDataError("No air quality data was returned for this location.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns=COLUMN_MAP)
    df["CO"] = df["CO"] / 1000.0  # µg/m³ -> mg/m³, to match the CPCB-trained model

    # Open-Meteo's hourly timestamps are local wall-clock time for the
    # queried coordinate (because timezone="auto"). Use the returned
    # UTC offset to compute "now" in that same local frame, rather than
    # relying on the machine running this code being in the same zone.
    utc_offset = payload.get("utc_offset_seconds", 0)
    local_now = pd.Timestamp.utcnow().tz_localize(None) + timedelta(seconds=utc_offset)
    df["is_forecast"] = df["time"] > local_now

    df = df.dropna(subset=["PM2.5", "PM10"])
    return df.reset_index(drop=True)


def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch contextual weather from Open-Meteo's weather endpoint."""
    try:
        response = requests.get(
            WEATHER_URL,
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation", "timezone": "auto"},
            timeout=10,
        )
        response.raise_for_status()
        current = response.json().get("current")
    except (requests.RequestException, ValueError) as exc:
        raise LiveDataError("Couldn't reach the weather service.") from exc
    if not current:
        raise LiveDataError("No weather data was returned for this location.")
    return current


def add_predictions(df: pd.DataFrame, reg, features: list) -> pd.DataFrame:
    """Predict AQI, then derive its CPCB category from that displayed value."""
    X = df[features]
    df = df.copy()
    # AQI is presented as a whole number throughout the UI.
    # Categorise that same value so a threshold cannot appear contradictory.
    df["predicted_aqi"] = reg.predict(X).round()
    df["predicted_bucket"] = df["predicted_aqi"].apply(aqi_to_bucket)
    return df


def daily_forecast_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    """Summarize hourly predictions and pollutant levels into daily rows for display."""
    columns = [
        "Date", "AQI average", "AQI peak", "Category",
        "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)",
        "SO2 (µg/m³)", "O3 (µg/m³)", "CO (mg/m³)",
    ]
    if forecast.empty:
        return pd.DataFrame(columns=columns)

    daily = (
        forecast.assign(Date=forecast["time"].dt.date)
        .groupby("Date", as_index=False)
        .agg(
            **{
                "AQI average": ("predicted_aqi", "mean"),
                "AQI peak": ("predicted_aqi", "max"),
                "PM2.5 (µg/m³)": ("PM2.5", "mean"),
                "PM10 (µg/m³)": ("PM10", "mean"),
                "NO2 (µg/m³)": ("NO2", "mean"),
                "SO2 (µg/m³)": ("SO2", "mean"),
                "O3 (µg/m³)": ("O3", "mean"),
                "CO (mg/m³)": ("CO", "mean"),
            }
        )
        .sort_values("Date")
    )
    daily["Category"] = daily["AQI average"].map(aqi_to_bucket)
    return daily[columns].reset_index(drop=True)


def pollutant_trend_summary(current: pd.Series, forecast: pd.DataFrame) -> pd.DataFrame:
    """Compare current concentrations with the next 24 hours, preserving units."""
    units = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³",
             "SO2": "µg/m³", "O3": "µg/m³", "CO": "mg/m³"}
    columns = ["Pollutant", "Current", "Next 24h avg", "Trend", "Unit"]
    if forecast.empty:
        return pd.DataFrame(columns=columns)
    horizon = forecast[forecast["time"] <= forecast["time"].min() + pd.Timedelta(hours=24)]
    rows = []
    for pollutant, unit in units.items():
        now_value = current.get(pollutant)
        values = horizon[pollutant].dropna()
        if pd.isna(now_value) or values.empty:
            direction, average = "Unavailable", None
        else:
            average = float(values.mean())
            delta = average - float(now_value)
            threshold = max(abs(float(now_value)) * 0.05, 0.01)
            direction = "Rising" if delta > threshold else "Falling" if delta < -threshold else "Stable"
        rows.append({"Pollutant": pollutant, "Current": now_value, "Next 24h avg": average,
                     "Trend": direction, "Unit": unit})
    return pd.DataFrame(rows, columns=columns)
