# Machine Learning Based Air Quality Prediction and Monitoring System

Worldwide pollutant lookup for **any location on Earth**, interpreted by a Random
Forest AQI model trained on 5 years of Indian CPCB station data (26 cities,
2015–2020) — plus daily AQI history for **292 Indian places in 32 states/UTs**
(Apr 2022 – Apr 2025), and a forecast of how AQI may move over the next few days.
The dashboard currently focuses on live estimates, an India map/place comparison,
and data-source status.

## Features

- **292 places, 2015–2025 history.** The Kaggle "India AQI" daily dataset
  (Apr 2022 – Apr 2025, 32 states/UTs) is merged with the original 26-city
  data. The Live tab includes a picker for Indian places, while India Map &
  Compare shows the latest available historical AQI and compares places.

- **Live data added.** [Open-Meteo](https://open-meteo.com) provides a
  completely free air-quality + geocoding API — no signup, no API key,
  no rate-limit token. Type any place name and the dashboard fetches its
  current atmospheric-model pollutant estimates and a multi-day forecast.
- **Forecasting.** Open-Meteo forecasts raw pollutant concentrations
  (PM2.5, PM10, CO, NO2, SO2, O3) for any location, but only computes
  US/European-style AQI. This project feeds those forecasted pollutant
  levels through our own CPCB-trained model to produce an **Indian-style
  AQI forecast** for the next several days — not just a snapshot of today.
- **Frontend redesign.** Custom-styled metric cards, colored AQI badges,
  an accessible light/dark appearance toggle, and interactive Plotly charts
  (hover tooltips, shaded forecast regions) make the dashboard easier to scan.
- **Forecast quality and usability.** The dashboard shows retrieval time,
  available forecast hours, missing-data caveats, pollutant trends, matured
  forecast comparisons, and a downloadable AQI/pollutant CSV.
- **Monitor validation and forecast diagnostics.** When a licensed monitor feed
  is configured, paired model/monitor snapshots are stored and summarized.
  Forecast revision MAE is also grouped by place and lead-time band.
- **Saved places and email alerts.** Save locations for the current browser
  session and optionally subscribe an email address to a daily AQI threshold
  alert from the scheduled collector.
- **Background collection.** A standalone collector can run on a schedule and
  write observations and forecast snapshots to SQLite, managed PostgreSQL, or
  managed MySQL.

## Project structure

```
air_quality_project/
├── data/
│   ├── city_day.csv            # CPCB 26 cities, 2015-2020, with pollutants (trains the model)
│   ├── aqi_daily_2022_2025.csv.gz  # Kaggle India AQI, 292 places, 2022-2025, daily AQI only
│   └── places.csv              # place -> state + map coordinates
├── models/                     # trained models saved here after training
├── outputs/                    # EDA plots saved here
├── data_prep_india.py          # loads/cleans/merges datasets and adds calendar flags
├── prepare_new_data.py         # one-time: raw Kaggle CSV -> the compact files above
├── live_data.py                # live + forecast pollutant data via Open-Meteo (no API key)
├── eda_india.py                # exploratory plots + crop-burning analysis
├── train_model_india.py        # trains the AQI regressor + writes evaluation metrics
├── dashboard.py                # Streamlit app (3 tabs, see below)
├── database.py                 # SQLite/PostgreSQL/MySQL storage and training data
├── forecast_tracker.py         # forecast-revision tracking in the shared database
├── collector.py                # scheduled, unattended forecast/data collection
├── notifications.py            # optional SMTP AQI alert emails
├── station_data.py             # optional approved official-monitor connector
├── tests/                      # pytest suite (no network / API key needed)
└── requirements.txt
```

SQLite is used by default for local development. Set `DATABASE_URL` to a hosted
PostgreSQL or MySQL connection URL to share data across the dashboard and collector.

## Setup

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

For the test suite, install the development dependencies:

```bash
pip install -r requirements-dev.txt
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
   - A map marker and the latest pollutant estimate
   - Our model's predicted AQI value + category (Good ... Severe) for
     right now
   - A chart of recent model estimates plus a several-day-ahead forecast,
     built by running Open-Meteo's forecasted pollutant levels through
     our trained model
   - A daily forecast table pairing average/peak model-predicted AQI and
     category with Open-Meteo's forecast PM2.5, PM10, NO2, SO2, O3 and CO levels
   - CSV download, pollutant concentration/trend breakdown, retrieval time,
     forecast coverage and missing-data caveats
   - Session-saved places, in-browser threshold alerts, scheduled email alerts,
     forecast-versus-later-model comparisons, and optional approved
     official-monitor comparison
2. **🗺️ India Map & Compare** — view latest historical AQI records by place
   and compare up to three places.
3. **ℹ️ Data Sources & Status** — data provenance, file freshness, model
   evaluation, monitor comparison MAE, and forecast revision MAE by place and
   forecast lead time.

## How the live prediction pipeline works

`live_data.py` calls two free Open-Meteo endpoints:

- **Geocoding** (`geocoding-api.open-meteo.com/v1/search`) — turns a
  typed place name into latitude/longitude.
- **Air Quality** (`air-quality-api.open-meteo.com/v1/air-quality`) —
  returns hourly PM2.5, PM10, CO, NO2, SO2, and O3 for the last couple of
  days plus several days ahead, for any coordinate on Earth.
- **Weather** (`api.open-meteo.com/v1/forecast`) — current temperature,
  humidity, wind and precipitation for context.

## Local database and retraining

The configured database stores bundled historical data, live pollutant
estimates, forecast snapshots, official-monitor readings when configured, and
model run metadata. SQLite is the default at `data/air_quality.sqlite`; set
`DATABASE_URL` for PostgreSQL or MySQL. The historical import runs once and is safe to repeat:

```bash
python -m database init
python -m database status
```

Live and forecast snapshots are retained for analysis and are explicitly
excluded from training because they are model outputs. The 2022–2025 history
contains AQI but no pollutant features, so it cannot train this pollutant-to-AQI
model. The bundled 2015–2020 pollutant/AQI data is training-eligible. To add
new pollutant measurements with verified AQI labels, use a CSV with columns
`City,Date,PM2.5,PM10,NO2,SO2,O3,CO,AQI` and document its provenance:

```bash
python -m database import-training-csv path/to/verified_data.csv --source "dataset and station provenance"
python train_model_india.py
```

Training reads only rows explicitly marked verified and eligible, writes the
regressor/evaluation report, saves a dated model artifact under `models/`, and
records the model version, features, data count and metrics in the database. Do not use
predicted AQI as a training label.

### Automatic collection and deployment

The dashboard can store lookups, but scheduled collection is handled separately
by `collector.py`; it does not depend on a person opening the dashboard. The
repository includes a GitHub Actions workflow that runs hourly and can also be
started manually. It is enabled after configuring the repository variables and
secrets described below.

For a small single-instance demo on a host that supports persistent disks, set
`AQI_DATABASE_PATH` to a file on the mounted disk (for example,
`/var/data/air_quality.sqlite`) and run just one app instance. Verify that the
host preserves that mounted path through restarts and deployments. Do not put
the database in an ephemeral application directory or run multiple app
instances against the same SQLite file.

For GitHub Actions, create a managed PostgreSQL or MySQL database, then add these in the
repository's Settings → Secrets and variables → Actions:

- Secret `DATABASE_URL`: the provider's connection URL. PostgreSQL commonly uses
  `postgresql://user:password@host:5432/database`; MySQL uses
  `mysql://user:password@host:3306/database`. Use the TLS/SSL settings required
  by your database host, URL-encode special characters in credentials, and keep
  this value private.
- Variable `AQI_COLLECTOR_LOCATIONS`: JSON array, for example
  `[ {"name":"Delhi","admin1":"Delhi","country":"India","latitude":28.6139,"longitude":77.209} ]`.
- Secrets `AQI_SMTP_HOST`, `AQI_SMTP_PORT`, `AQI_SMTP_USER`,
  `AQI_SMTP_PASSWORD`, and `AQI_ALERT_FROM` for scheduled alert email delivery.
- Secret `CPCB_AQI_URL` for an approved monitor endpoint if you want the
  scheduled collector to capture model/monitor pairs.

Saved places stay in the current Streamlit browser session. Email alert
subscriptions are stored in the shared database; the collector includes those
coordinates automatically and checks them hourly, sending at most one email per
subscription per UTC day after its threshold is crossed. Configure SMTP settings
as GitHub Actions secrets (and keep the dashboard connected to the same
database). Email ownership is not verified in this demo, so only subscribe an
address you control; put a public deployment behind authentication or add
verified-email signup before opening subscriptions to arbitrary visitors.

The official-monitor panel records paired snapshots when a licensed endpoint is
configured. Its MAE is a useful diagnostic, but time and distance differences
mean it is not a controlled model-validation study. The city/lead-time table is
forecast-to-later-model comparison, not ground-truth accuracy.

The first collector run initializes the schema and imports the bundled history.
To copy locally collected SQLite records as well, run
`python -m database migrate-sqlite data/air_quality.sqlite` with `DATABASE_URL`
pointing at the destination first. Keep database credentials out of source
control. Use one database for the dashboard and collector. Do not train on
Open-Meteo predictions; only independently verified observations with valid AQI
labels should be marked training-eligible.

The included workflow runs at minute 17 of every hour (GitHub schedule timing
can be delayed). For a different host, set the same environment variables and
schedule `python collector.py` with its cron or managed scheduler.

Those pollutant estimates (renamed and unit-converted — Open-Meteo reports
CO in µg/m³, but the CPCB training data uses mg/m³, so it's divided by
1000) are then passed through the trained `india_aqi_regressor.joblib` model.
The category is derived directly from the displayed AQI using CPCB thresholds,
so the number and label cannot disagree.

**Model performance** is assessed with a chronological holdout, which keeps
later dates out of training. Running `python train_model_india.py` writes MAE,
R² and CPCB-category accuracy to `outputs/model_evaluation.json`. It uses only
the 6 pollutants Open-Meteo can supply — PM2.5, PM10, NO2, SO2, O3, CO.

## Data notes (read before relying on the history)

- **Two sources, different content.** `city_day.csv` has pollutant
  concentrations, so it trains the model. The 2022–2025 Kaggle file has only a
  daily AQI per place and the dominant pollutant(s), so it **cannot** retrain the
  pollutant→AQI model; it feeds the India map/comparison view and live-tab historical context.
- **Dates.** The Kaggle file runs 1 Apr 2022 – 30 Apr 2025 (not 2023–2025), and
  there is **no data from Jul 2020 to Mar 2022** between the historical sources.
  Crop-burning windows are covered for 2015–2019 (few cities early on) and 2022–2024.
- **Uneven coverage.** Some places have under a year of data; comparisons should
  be interpreted with the number of available days in mind.
- **Same name, two places.** "Aurangabad" exists in Bihar and Maharashtra; they
  appear as `Aurangabad (Bihar)` / `Aurangabad (Maharashtra)`.
- **Map coordinates** for the extra places come from the offline GeoNames city
  list ([geonames.org](https://www.geonames.org), CC-BY 4.0), 19 are hand-entered
  approximations, and 11 industrial-area stations have no pin. Rebuild with
  `python prepare_new_data.py <raw csv>` (needs `pip install geonamescache`).
- **Licence.** Check the Kaggle dataset's licence before redistributing the
  data in a public repo. The underlying figures are CPCB's.

## Notes and honest caveats

- **AQI thresholds**: India's official CPCB breakpoints are used (Good
  ≤50, Satisfactory ≤100, Moderate ≤200, Poor ≤300, Very Poor ≤400,
  Severe >400), applied to the model's numeric AQI prediction.
- **Coordinates** (the original 26 hardcoded, the rest from GeoNames) are
  for dropping a map pin, not survey-grade.
- **The live tab needs internet access** to reach Open-Meteo at runtime;
  if the request fails (no connection, location not found, etc.) the
  dashboard shows a plain error message rather than crashing.
- **Forecast accuracy compounds**: the further out the forecast, the more
  uncertain Open-Meteo's own pollutant forecast is, and our model adds
  its own error on top of that — treat the multi-day forecast as a
  directional trend, not a precise reading.
- **Live-data limitation.** Open-Meteo supplies atmospheric-model estimates
  rather than local regulatory-monitor observations, and our model was
  trained on daily station averages before being applied to hourly modelled data.
  Treat live AQI values as estimates.
