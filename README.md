# Machine Learning Based Air Quality Prediction and Monitoring System

Live air quality lookup for **any location on Earth**, scored by a Random
Forest model trained on 5 years of Indian CPCB station data (26 cities,
2015–2020) — plus daily AQI history for **292 Indian places in 32 states/UTs**
(Apr 2022 – Apr 2025), plus a forecast of how AQI will likely move over the next
few days, and a dedicated analysis of the annual crop-burning-season
spike in North India.

## What changed from the first version

- **292 places, 2015–2025 history (latest update).** The Kaggle "India AQI"
  daily dataset (Apr 2022 – Apr 2025, 32 states/UTs) is merged with the
  original 26-city data. The History Explorer has a state filter and all 292
  places, the Crop Burning tab adds season-by-season and state-level views for
  2022–2024, the Live tab has an "Indian place" picker, and the assistant has a
  new `rank_places` tool ("most polluted places in Punjab").

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
- **AQI Assistant (new).** A chat tab where an LLM (Claude) answers
  questions like "Is it safe for my asthmatic father to walk in Lucknow
  tomorrow morning?" by calling this project's own code as tools — live
  AQI, forecast, city history, crop-burning impact and health guidance —
  instead of guessing. See [The AQI Assistant](#the-aqi-assistant-llm--tool-calling).

## Project structure

```
air_quality_project/
├── data/
│   ├── city_day.csv            # CPCB 26 cities, 2015-2020, with pollutants (trains the model)
│   ├── aqi_daily_2022_2025.csv.gz  # Kaggle India AQI, 292 places, 2022-2025, daily AQI only
│   └── places.csv              # place -> state + map coordinates
├── models/                     # trained models saved here after training
├── outputs/                    # EDA plots saved here
├── data_prep_india.py          # loads/cleans/merges the datasets, crop-burning flag, search
├── prepare_new_data.py         # one-time: raw Kaggle CSV -> the compact files above
├── live_data.py                # live + forecast pollutant data via Open-Meteo (no API key)
├── eda_india.py                # exploratory plots + crop-burning analysis
├── train_model_india.py        # trains the AQI regressor + classifier
├── assistant_tools.py          # the tools the chat assistant can call (+ their JSON schemas)
├── assistant.py                # LLM tool-calling loop (Anthropic API)
├── dashboard.py                # Streamlit app (4 tabs, see below)
├── tests/                      # pytest suite (no network / API key needed)
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

## The dashboard's 4 tabs

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
3. **📈 City History Explorer** — filter by state, search any of the 292
   places, and see its daily AQI (2015–2020 and 2022–2025), map pin, crop-season
   effect and most common dominant pollutant.
4. **🤖 AQI Assistant** — chat with an LLM that answers by calling the
   project's own tools. Needs an Anthropic API key (sidebar); the other
   three tabs work without one.

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

## The AQI Assistant (LLM + tool calling)

The fourth tab is a chat assistant. The LLM never makes up air-quality
numbers: to answer, it has to call one of the tools below, which return real
data (or a clear error). Expand **"🔧 tool calls used"** under any answer to
see exactly which tools ran and what they returned.

```
question → Claude (Anthropic API) → chooses tool(s) → assistant_tools.py runs them
         → results go back to Claude → plain-language answer
```

| Tool | What it does | Data behind it |
|---|---|---|
| `get_current_aqi(place)` | Estimated AQI, category and pollutant levels right now, for any place | Open-Meteo + our model |
| `get_aqi_forecast(place, days)` | Per-day average/peak AQI, worst and cleanest hour, improving/worsening trend | Open-Meteo + our model |
| `get_city_history(city)` | Summary for any of the 292 places (aliases like Bangalore/Gurgaon work) | merged CSVs |
| `compare_cities(a, b)` | Side-by-side history, incl. crop-burning season | merged CSVs |
| `get_crop_burning_impact()` | In-season vs rest-of-year AQI, most affected cities | merged CSVs |
| `rank_places(metric, state, order, limit)` | Most/least polluted places, nationally or within a state | merged CSVs |
| `get_health_guidance(category, profile)` | Possible impacts and precautions by AQI category and person type (asthma, child, runner, ...) | built-in text |

**API key** — any one of:
1. paste it in the sidebar box (kept only in your browser session), or
2. set the `ANTHROPIC_API_KEY` environment variable, or
3. put `ANTHROPIC_API_KEY = "sk-ant-..."` in `.streamlit/secrets.toml`.

Usage is billed per Anthropic's API pricing; the default model (Claude Haiku
5.5) is the cheaper option, Claude Sonnet 5.5 can be picked in the sidebar.

**Design choices**
- **Model:** Claude through the official `anthropic` SDK using native tool
  use. The model is a setting, so it is easy to swap.
- **No agent framework:** the loop in `assistant.py` is ~20 lines (call the
  model → run requested tools → send results back → repeat, max 6 steps).
  With a small, fixed tool set that is easier to explain and debug than a
  framework; LangGraph or similar could be added later.
- **No database:** tools read the CSV and the saved models, live data is cached
  in memory for 15 minutes, chat history lives in the browser session only.
  A database would be the next step if you want to store readings over time.
- **Grounding and safety:** tool failures come back as data (so the chat never
  crashes), the system prompt forbids quoting numbers from memory, health text
  is fixed rather than improvised, and every answer carries a not-medical-advice
  note.

**Tests** — `pip install pytest`, then `python -m pytest tests -q` (29 tests).
They need no internet and no API key: live Open-Meteo calls are replaced with
fake data and the Anthropic API with a scripted fake client, while the real
models, CPCB data, tools and dashboard are exercised. They therefore verify
the loop, tools and UI, but **not** a real round-trip to the Anthropic API;
try that yourself with your key.

## Data notes (read before relying on the history)

- **Two sources, different content.** `city_day.csv` has pollutant
  concentrations, so it trains the model. The 2022–2025 Kaggle file has only a
  daily AQI per place and the dominant pollutant(s), so it **cannot** retrain the
  pollutant→AQI model; it feeds the history, crop-burning and ranking views.
- **Dates.** The Kaggle file runs 1 Apr 2022 – 30 Apr 2025 (not 2023–2025), and
  there is **no data from Jul 2020 to Mar 2022** — charts show that as a break.
  Crop-burning windows are covered for 2015–2019 (few cities early on) and 2022–2024.
- **Uneven coverage.** Some places have under a year of data; rankings only
  include places with ≥365 days (≥30 in-season days for season rankings).
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
- **The assistant is an LLM**: the numbers it quotes come from the tools, but
  the wording is the model's, so it can still misread a question or summarise
  badly. Its health advice is general information, not medical advice. The
  per-category health text in `assistant_tools.py` is a simplified summary in
  the spirit of the CPCB AQI categories, not a verbatim quote — check the
  official CPCB document before citing it.
- **Same caveat applies to its live answers**: the model was trained on daily
  station averages and is applied to hourly modelled data, so treat live AQI
  values as estimates.
