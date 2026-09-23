# Machine Learning Based Air Quality Prediction and Monitoring System

Live air quality lookup for **any location on Earth**, scored by a Random
Forest model trained on 5 years of Indian CPCB station data (26 cities,
2015–2020) — plus a forecast of how AQI will likely move over the next
few days, and a dedicated analysis of the annual crop-burning-season
spike in North India.

## What changed from the first version

- **Italy dataset removed.** The project is now India-focused end to end.
- **Live data added.** [Open-Meteo](https://open-meteo.com) provides a
  completely free air-quality + geocoding API — no signup, no API key,
  no rate-limit token. Type any place name and the dashboard fetches its
  real current pollutant readings and a multi-day forecast.
- **Forecasting.** Open-Meteo forecasts raw pollutant concentrations
  (PM2.5, PM10, CO, NO2, SO2, O3) for any location, but only computes
  US/European-style AQI. This project feeds those forecasted pollutant
  levels through our own CPCB-trained model to produce an **Indian-style
  AQI forecast** for the next several days — not just a snapshot of today.
- **Frontend redesign.** Custom-styled metric cards, colored AQI badges,
  and interactive Plotly charts (hover tooltips, shaded forecast regions)
  replace the earlier static matplotlib images and plain `st.metric` calls.

## Project structure

```
air_quality_project/
├── data/
│   └── city_day.csv            # historical India dataset (26 cities, 2015-2020)
├── models/                     # trained models saved here after training
├── outputs/                    # EDA plots saved here
├── data_prep_india.py          # loads/cleans the historical CSV, crop-burning flag, city search
├── live_data.py                # live + forecast pollutant data via Open-Meteo (no API key)
├── eda_india.py                # exploratory plots + crop-burning analysis
├── train_model_india.py        # trains the AQI regressor + classifier
├── dashboard.py                # Streamlit app (3 tabs, see below)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Sanity-check the historical data
python3 data_prep_india.py

# 2. Generate EDA plots (saved to outputs/)
python3 eda_india.py

# 3. Train the models (saved to models/)
python3 train_model_india.py

# 4. Launch the dashboard
streamlit run dashboard.py
```

## The dashboard's 3 tabs

1. **🌍 Live Air Quality & Forecast** — type any place name (not limited
   to India — Open-Meteo's geocoding covers the whole world). You get:
   - A map marker and the live current pollutant readings
   - Our model's predicted AQI value + category (Good ... Severe) for
     right now
   - A chart of recent actual AQI plus a several-day-ahead forecast,
     built by running Open-Meteo's forecasted pollutant levels through
     our trained model
   - A pollutant breakdown bar chart (PM2.5, PM10, NO2, SO2, O3, CO)
2. **🔥 Crop Burning Season Impact** — the historical, dataset-driven
   view of how much AQI rises in North Indian cities during the
   mid-October to end-November stubble-burning window.
3. **📈 City History Explorer** — browse the full 2015–2020 daily AQI
   record for any of the 26 cities in the training data.

## How the live prediction pipeline works

`live_data.py` calls two free Open-Meteo endpoints:

- **Geocoding** (`geocoding-api.open-meteo.com/v1/search`) — turns a
  typed place name into latitude/longitude.
- **Air Quality** (`air-quality-api.open-meteo.com/v1/air-quality`) —
  returns hourly PM2.5, PM10, CO, NO2, SO2, and O3 for the last couple of
  days plus several days ahead, for any coordinate on Earth.

Those pollutant readings (renamed and unit-converted — Open-Meteo reports
CO in µg/m³, but the CPCB training data uses mg/m³, so it's divided by
1000) are then passed through the trained `india_aqi_regressor.joblib`
and `india_aqi_classifier.joblib` models to get a predicted AQI value and
category for every hour, actual and forecast alike.

**Model performance** (on held-out historical data, using only the 6
pollutants Open-Meteo can supply — PM2.5, PM10, NO2, SO2, O3, CO):
- AQI category classifier: **~79% accuracy**
- Numeric AQI regressor: **R² ≈ 0.90**

## Notes and honest caveats

- **AQI thresholds**: India's official CPCB breakpoints are used (Good
  ≤50, Satisfactory ≤100, Moderate ≤200, Poor ≤300, Very Poor ≤400,
  Severe >400), applied to the model's numeric AQI prediction.
- **City coordinates** in the History Explorer tab are hardcoded
  approximate city-center lat/lon (not part of the original dataset) —
  fine for dropping a map pin, not survey-grade.
- **The live tab needs internet access** to reach Open-Meteo at runtime;
  if the request fails (no connection, location not found, etc.) the
  dashboard shows a plain error message rather than crashing.
- **Forecast accuracy compounds**: the further out the forecast, the more
  uncertain Open-Meteo's own pollutant forecast is, and our model adds
  its own error on top of that — treat the multi-day forecast as a
  directional trend, not a precise reading.
