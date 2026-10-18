"""
prepare_new_data.py
-------------------
ONE-TIME helper that turns the raw Kaggle "India AQI 2022-2025" CSV (a ~30 MB
file with one row per state/area/day) into the two small files the app reads:

    data/aqi_daily_2022_2025.csv.gz   date, state, place, stations, dominant_pollutants, aqi
    data/places.csv                   place, state, latitude, longitude, coord_source

You only need to run this if you want to rebuild those files:

    pip install geonamescache
    python prepare_new_data.py path/to/aqi.csv

Coordinates come from the offline GeoNames city list (`geonamescache`, CC-BY 4.0,
https://www.geonames.org). Places GeoNames can't match by name get a hand-entered
approximate coordinate where I was confident, otherwise no coordinate (no map pin).
Coordinates are for dropping a map pin only, not survey-grade.
"""

import sys

import pandas as pd

from data_prep_india import CITY_COORDS

# GeoNames admin1 codes for India, used to pick the right same-named city.
STATE_ADMIN1 = {
    "Andaman and Nicobar Islands": "01", "Andhra Pradesh": "02", "Assam": "03",
    "Chandigarh": "05", "Delhi": "07", "Gujarat": "09", "Haryana": "10",
    "Himachal Pradesh": "11", "Jammu and Kashmir": "12", "Kerala": "13",
    "Maharashtra": "16", "Manipur": "17", "Meghalaya": "18", "Karnataka": "19",
    "Nagaland": "20", "Odisha": "21", "Puducherry": "22", "Punjab": "23",
    "Rajasthan": "24", "West Bengal": "28", "Tamil Nadu": "25", "Tripura": "26", "Sikkim": "29",
    "Arunachal Pradesh": "30", "Mizoram": "31", "Bihar": "34",
    "Madhya Pradesh": "35", "Uttar Pradesh": "36", "Chhattisgarh": "37",
    "Jharkhand": "38", "Uttarakhand": "39", "Telangana": "40",
}

# Approximate coordinates for places GeoNames could not match by name.
# Only places I was confident about; the rest stay without a map pin.
MANUAL_COORDS = {
    "Byrnihat": (26.04, 91.87), "Mira Bhayandar": (19.29, 72.87),
    "Vatva": (22.97, 72.65), "Chikkaballapur": (13.43, 77.73),
    "Belapur": (19.02, 73.04), "Davanagere": (14.46, 75.92),
    "Dharwad": (15.46, 75.01), "Jalore": (25.35, 72.62),
    "Gummidipoondi": (13.41, 80.11), "Sivasagar": (26.98, 94.64),
    "Ariyalur": (11.14, 79.08), "Ramanagara": (12.72, 77.28),
    "Mandi Gobindgarh": (30.67, 76.30), "Virudhunagar": (9.58, 77.96),
    "Nandesari": (22.33, 73.09), "Ballabgarh": (28.34, 77.32),
    "Kurukshetra": (29.97, 76.88), "Gandhi Nagar": (23.22, 72.65), "Jorapokhar": CITY_COORDS["Jorapokhar"],
}


def build_places(df):
    import geonamescache

    gc = geonamescache.GeonamesCache()
    cities = {c["geonameid"]: c for c in gc.get_cities().values() if c["countrycode"] == "IN"}
    by_name = {}
    for c in cities.values():
        for n in set([c["name"]] + c.get("alternatenames", [])):
            by_name.setdefault(n.lower(), []).append(c)

    rows = []
    for place, state in df[["place", "state"]].drop_duplicates().itertuples(index=False):
        base = place.split(" (")[0]
        cands = by_name.get(base.lower(), [])
        in_state = [c for c in cands if c["admin1code"] == STATE_ADMIN1.get(state)]
        lat = lon = None
        src = ""
        if in_state:
            c = max(in_state, key=lambda c: c["population"])
            lat, lon, src = c["latitude"], c["longitude"], "GeoNames"
        elif base in MANUAL_COORDS:
            lat, lon = MANUAL_COORDS[base]
            src = "approximate (manual)"
        rows.append({"place": place, "state": state, "latitude": lat, "longitude": lon, "coord_source": src})
    return pd.DataFrame(rows).sort_values("place")


def main(raw_path):
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df = df.rename(columns={
        "area": "place", "number_of_monitoring_stations": "stations",
        "prominent_pollutants": "dominant_pollutants", "aqi_value": "aqi",
    })
    # Two different places are both called "Aurangabad" (Bihar, Maharashtra)
    names = df.groupby("place")["state"].nunique()
    dup = set(names[names > 1].index)
    df["place"] = [f"{p} ({s})" if p in dup else p for p, s in zip(df["place"], df["state"])]

    df = df.drop_duplicates(["date", "place"])
    keep = ["date", "state", "place", "stations", "dominant_pollutants", "aqi"]
    df[keep].sort_values(["place", "date"]).to_csv(
        "data/aqi_daily_2022_2025.csv.gz", index=False, date_format="%Y-%m-%d")

    places = build_places(df)
    places.to_csv("data/places.csv", index=False)
    print(f"{len(df):,} rows, {df['place'].nunique()} places, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    print(f"coordinates: {places['latitude'].notna().sum()} / {len(places)} places")
    print("no coordinate:", places[places["latitude"].isna()]["place"].tolist())


if __name__ == "__main__":
    main(sys.argv[1])
