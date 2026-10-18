"""
data_prep_india.py
-------------------
Loads and cleans the India air quality data:
  * city_day.csv  -- 26 cities, daily pollutants + AQI, 2015-2020 (CPCB station data,
    the well-known Kaggle "Air Quality Data in India" set). Trains the model.
  * aqi_daily_2022_2025.csv.gz -- 292 places across 32 states/UTs, daily AQI only,
    Apr 2022 - Apr 2025 (Kaggle India AQI dataset; see prepare_new_data.py).
    Used for history, crop-burning analysis and the assistant.

Adds two things the Italy dataset didn't need:
  1. A location/city search helper, since this dataset covers 26 cities.
  2. A "crop burning season" flag. Every year around mid-October to the
     end of November, large-scale paddy stubble burning across Punjab and
     Haryana sends a smoke plume across North India, driving a sharp,
     well-documented AQI spike in Delhi-NCR and nearby cities (Amritsar,
     Chandigarh, Lucknow, Patna). Flagging this window lets the dashboard
     show its effect directly instead of it just being lost in the noise
     of "October" and "November" as ordinary months.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/city_day.csv"
NEW_PATH = "data/aqi_daily_2022_2025.csv.gz"
PLACES_PATH = "data/places.csv"

# Approximate city-center coordinates for the 26 cities in the dataset.
# Used only to place a marker on the dashboard map -- not part of the
# original dataset (which has no lat/lon column).
CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5062, 80.6480), "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167), "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.7041, 77.1025), "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266), "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867), "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7104, 86.4131), "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777), "Patna": (25.5941, 85.1376),
    "Shillong": (25.5788, 91.8933), "Talcher": (20.9500, 85.2167),
    "Thiruvananthapuram": (8.5241, 76.9366), "Visakhapatnam": (17.6868, 83.2185),
}

# Cities where the stubble-burning plume from Punjab/Haryana has the most
# documented impact -- used to highlight the "most affected" view.
CROP_BURNING_HOTSPOT_CITIES = ["Delhi", "Amritsar", "Chandigarh", "Lucknow", "Gurugram", "Patna"]

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene",
]


def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])

    # Xylene is missing for >60% of rows across the whole dataset -- same
    # situation as NMHC(GT) in the Italy data, so it's dropped rather than
    # imputed into meaninglessness.
    df = df.drop(columns=["Xylene"], errors="ignore")

    # Fill gaps per city using time-ordered interpolation (a city's own
    # recent readings are a far better stand-in than another city's, or
    # the dataset-wide mean) then drop whatever's still missing.
    df = df.sort_values(["City", "Date"])
    fill_cols = [c for c in POLLUTANT_COLS if c in df.columns] + ["AQI"]
    df[fill_cols] = (
        df.groupby("City")[fill_cols]
        .apply(lambda g: g.interpolate(limit_direction="both"))
        .reset_index(level=0, drop=True)
    )
    # A handful of cities (e.g. Lucknow) have a pollutant column that's
    # 100% missing, which per-city interpolation can't fix -- fall back to
    # the dataset-wide median for those so the city isn't dropped entirely.
    for col in fill_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["AQI"])

    # Recompute AQI_Bucket wherever it's missing but AQI is now known
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna(df["AQI"].apply(_aqi_to_bucket))

    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year

    # Crop burning season: mid-October through end of November each year
    df["is_crop_burning_season"] = (
        ((df["month"] == 10) & (df["day"] >= 15)) | (df["month"] == 11)
    )

    return df.reset_index(drop=True)


# State for each of the 26 original cities (the old CSV has no state column).
OLD_CITY_STATE = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Andhra Pradesh",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Patna": "Bihar", "Shillong": "Meghalaya", "Talcher": "Odisha",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}


def _add_calendar(df):
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["year"] = df["Date"].dt.year
    df["is_crop_burning_season"] = (
        ((df["month"] == 10) & (df["day"] >= 15)) | (df["month"] == 11)
    )
    return df


def load_new_data(path: str = NEW_PATH) -> pd.DataFrame:
    """Kaggle India AQI 2022-2025: daily AQI per place (no pollutant concentrations)."""
    d = pd.read_csv(path, parse_dates=["date"])
    d = d.rename(columns={"place": "City", "state": "State", "date": "Date", "aqi": "AQI"})
    d["AQI_Bucket"] = d["AQI"].apply(_aqi_to_bucket)  # CPCB breakpoints, same as the old data
    d["source"] = "2022-2025"
    return _add_calendar(d)


def load_history(old_path: str = RAW_PATH, new_path: str = NEW_PATH) -> pd.DataFrame:
    """
    Both datasets stacked into one daily-AQI history (columns: City, State, Date,
    AQI, AQI_Bucket, source, dominant_pollutants, stations + calendar flags).
    Used for the history/crop-burning views and the assistant. The pollutant
    model is still trained on the old data only (the new file has no pollutants).
    There is a gap between Jul 2020 and Mar 2022 where neither file has data.
    """
    old = load_and_clean(old_path)[["City", "Date", "AQI", "AQI_Bucket"]].copy()
    old["State"] = old["City"].map(OLD_CITY_STATE)
    old["source"] = "2015-2020"
    old["dominant_pollutants"] = None
    old["stations"] = pd.NA
    old = _add_calendar(old)
    new = load_new_data(new_path)
    cols = ["City", "State", "Date", "AQI", "AQI_Bucket", "source",
            "dominant_pollutants", "stations", "month", "day", "year", "is_crop_burning_season"]
    df = pd.concat([old[cols], new[cols]], ignore_index=True)
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def load_places(path: str = PLACES_PATH) -> pd.DataFrame:
    """place -> state, latitude, longitude (coordinates may be missing)."""
    p = pd.read_csv(path).rename(columns={"place": "City", "state": "State"})
    old = pd.DataFrame(
        [{"City": c, "State": OLD_CITY_STATE[c], "latitude": lat, "longitude": lon,
          "coord_source": "hardcoded"} for c, (lat, lon) in CITY_COORDS.items()])
    # the 26 original hand-set coordinates win over GeoNames for those cities
    p = p[~p["City"].isin(old["City"])]
    return pd.concat([old, p], ignore_index=True).sort_values("City").reset_index(drop=True)


def place_coords(places, city):
    """(lat, lon) for a city, or None if we have no coordinate for it."""
    row = places[places["City"] == city]
    if row.empty or pd.isna(row.iloc[0]["latitude"]):
        return None
    return float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"])


def _aqi_to_bucket(aqi: float) -> str:
    """Standard Indian CPCB AQI category breakpoints."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def search_city(query: str, cities) -> list:
    """Case-insensitive substring match against the list of known cities."""
    query = query.strip().lower()
    if not query:
        return []
    return [c for c in cities if query in c.lower()]


if __name__ == "__main__":
    data = load_and_clean()
    print(f"Cleaned shape: {data.shape}")
    print(f"Cities: {sorted(data['City'].unique())}")
    print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"\nCrop-burning-season rows: {data['is_crop_burning_season'].sum()}")
    print("\nAvg AQI: crop-burning season vs rest of year (hotspot cities)")
    hotspot = data[data["City"].isin(CROP_BURNING_HOTSPOT_CITIES)]
    print(hotspot.groupby("is_crop_burning_season")["AQI"].mean())
