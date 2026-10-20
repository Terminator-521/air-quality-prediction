"""
dashboard.py
------------
Streamlit dashboard for the Air Quality Prediction project.

Run with:
    streamlit run dashboard.py

Three tabs:
  1. Live Air Quality & Forecast -- search ANY place on Earth (via
     Open-Meteo's free geocoding API), pull its live + 4-day-forecast
     pollutant readings (via Open-Meteo's free air-quality API), and run
     them through our own Random Forest model (trained on
     Indian CPCB station data) to get a predicted Indian-style AQI value
     and category for right now AND for the next few days. Open-Meteo
     gives raw pollutant forecasts but only computes US/European AQI --
     this is what turns that into an Indian CPCB-style AQI forecast.
  2. India Map & Compare -- compare historical AQI across places.
  3. Data Sources & Status -- provenance, freshness and monitor connection.
"""

import html
import json
from datetime import datetime, timezone
from io import StringIO

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, DATABASE_URL, LIVE_CACHE_SECONDS, MODELS_DIR, OUTPUTS_DIR
from data_prep_india import (
    load_history, load_places, place_coords,
)
from live_data import (
    geocode, fetch_air_quality, fetch_weather, add_predictions,
    daily_forecast_summary, pollutant_trend_summary, LiveDataError,
)
from forecast_tracker import (
    record as record_forecast, summary as forecast_summary,
    accuracy_by_location as forecast_accuracy_by_location,
)
from station_data import OfficialMonitorError, configured as official_monitor_configured, fetch_nearest_monitor
import database

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="wide")

# ---------------------------------------------------------------- styling
BUCKET_COLORS = {
    "Good": "#0072B2", "Satisfactory": "#56B4E9", "Moderate": "#009E73",
    "Poor": "#E69F00", "Very Poor": "#D55E00", "Severe": "#CC79A7",
}
BUCKET_TEXT = {
    "Good": "white", "Satisfactory": "#1a1a1a", "Moderate": "white",
    "Poor": "#1a1a1a", "Very Poor": "white", "Severe": "white",
}

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: #f4f7fb; color: #17243a; }
[data-testid="stAppViewContainer"] .main { background: #f4f7fb; }
.block-container { max-width: 1440px; padding: 1.35rem 2.2rem 3rem; }
.hero {
    background: linear-gradient(115deg, #102945 0%, #174a70 55%, #167d83 100%);
    padding: 2rem 2.25rem; border-radius: 22px; color: #fff; margin: .65rem 0 1.5rem;
    box-shadow: 0 16px 34px rgba(20, 54, 82, .14); position: relative; overflow: hidden;
}
.hero:after { content: ""; position: absolute; width: 250px; height: 250px; border-radius: 50%; right: 7%; top: -145px; background: rgba(255,255,255,.08); }
.hero .eyebrow { margin: 0 0 .5rem; color: #9ddbd5; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
.hero h1 { margin: 0; font-size: clamp(1.65rem, 3vw, 2.35rem); line-height: 1.15; font-weight: 750; letter-spacing: -.035em; }
.hero p { margin: .65rem 0 0; max-width: 780px; color: #d9e8f1; font-size: .96rem; line-height: 1.6; }
.section-kicker { color: #53718c; font-size: .72rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .2rem; }
.section-title { color: #17243a; font-size: 1.34rem; font-weight: 700; letter-spacing: -.025em; margin: 0 0 .3rem; }
.section-help { color: #687a8e; font-size: .9rem; margin: 0 0 .8rem; }
.metric-card {
    background: #fff; border: 1px solid #e4ebf2; border-radius: 16px; padding: 1rem 1.1rem;
    border-top: 3px solid #168b8b; min-height: 106px; box-shadow: 0 4px 14px rgba(23, 48, 72, .035);
}
.metric-card .label { color: #6d7d90; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .075em; }
.metric-card .value { color: #17243a; font-size: 1.55rem; font-weight: 750; margin-top: .28rem; letter-spacing: -.035em; }
.metric-card .sub { color: #748398; font-size: .76rem; margin-top: .25rem; line-height: 1.4; }
.pollutant-card { background: #fff; border: 1px solid #e4ebf2; border-radius: 15px; padding: .9rem 1rem; margin-bottom: .65rem; min-height: 104px; }
.pollutant-card .pollutant-name { color: #718198; font-size: .72rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
.pollutant-card .pollutant-value { color: #17243a; font-size: 1.32rem; font-weight: 750; margin: .32rem 0 .15rem; }
.pollutant-card .pollutant-trend { color: #63768a; font-size: .75rem; }
.aqi-badge {
    display: inline-block; padding: 6px 12px; border-radius: 999px;
    font-weight: 700; font-size: .85rem; letter-spacing: .015em;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid #e1e8ef; }
[data-testid="stTabs"] button[data-baseweb="tab"] { color: #64768a; border-radius: 10px 10px 0 0; padding: .7rem .95rem; font-weight: 650; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #126f75; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #168b8b; height: 3px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #e4ebf2; border-radius: 16px; }
[data-testid="stDataFrame"] { border: 1px solid #e4ebf2; border-radius: 12px; overflow: hidden; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e4ebf2; border-radius: 14px; padding: .85rem 1rem; }
div[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 650; min-height: 2.6rem; }
@media (max-width: 800px) { .block-container { padding: .85rem 1rem 2rem; } .hero { padding: 1.5rem; border-radius: 17px; } }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'


def section_heading(title, helper=None, kicker=None):
    if kicker:
        st.markdown(f'<div class="section-kicker">{html.escape(str(kicker))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{html.escape(str(title))}</div>', unsafe_allow_html=True)
    if helper:
        st.markdown(f'<div class="section-help">{html.escape(str(helper))}</div>', unsafe_allow_html=True)


def badge_html(bucket, aqi=None):
    bg = BUCKET_COLORS.get(bucket, "#999")
    fg = BUCKET_TEXT.get(bucket, "white")
    val = f" · AQI {aqi:.0f}" if aqi is not None else ""
    return f'<span class="aqi-badge" style="background:{bg};color:{fg};">{bucket}{val}</span>'


# ---------------------------------------------------------------- cached loaders
@st.cache_resource
def load_models():
    reg = joblib.load(MODELS_DIR / "india_aqi_regressor.joblib")
    features = joblib.load(MODELS_DIR / "india_features.joblib")
    return reg, features

@st.cache_data
def load_historical():
    return load_history()

@st.cache_data
def load_place_table():
    return load_places()


@st.cache_data
def load_evaluation():
    path = OUTPUTS_DIR / "model_evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(ttl=LIVE_CACHE_SECONDS)
def cached_geocode(query):
    return geocode(query)

@st.cache_data(ttl=LIVE_CACHE_SECONDS)
def cached_air_quality(lat, lon):
    return fetch_air_quality(lat, lon), datetime.now(timezone.utc)


@st.cache_data(ttl=LIVE_CACHE_SECONDS)
def cached_weather(lat, lon):
    return fetch_weather(lat, lon)


@st.cache_data(ttl=LIVE_CACHE_SECONDS)
def cached_official_monitor(lat, lon):
    return fetch_nearest_monitor(lat, lon)


reg_model, FEATURES = load_models()
hist_df = load_historical()
places_df = load_place_table()
evaluation = load_evaluation()
database.import_project_history()
database.migrate_forecast_tracker()
HIST_CITIES = sorted(hist_df["City"].unique())
st.session_state.setdefault("in_app_alerts", {})
st.session_state.setdefault("saved_locations", [])

hero_left, hero_controls = st.columns([5, 2])
mode_control, text_control = hero_controls.columns(2, gap="small")
dark_mode = mode_control.toggle("Dark mode", key="dark_mode", help="Switch the dashboard between light and dark appearance.")
large_text = text_control.toggle("Large text", key="large_text", help="Increase text size throughout the dashboard.")
font_scale = "1.08rem" if large_text else "1rem"
st.markdown(f"<style>[data-testid='stAppViewContainer'] {{ font-size: {font_scale}; }}</style>", unsafe_allow_html=True)
if dark_mode:
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] .main,
    [data-testid="stHeader"], [data-testid="stBottom"] { background: #0d1622 !important; color: #e6edf5 !important; }
    .block-container { color: #e6edf5; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { color: #e6edf5; }
    .hero p { color: #d9e8f1 !important; }
    .section-kicker { color: #8aa1b8; }
    .section-title { color: #edf3f9; }
    .section-help, [data-testid="stCaptionContainer"] { color: #a0afc0 !important; }
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"], .metric-card, .pollutant-card {
        background: #152231 !important; border-color: #2a3a4c !important; color: #e6edf5 !important;
    }
    .metric-card .label, .pollutant-card .pollutant-name { color: #a5b4c5 !important; }
    .metric-card .value, .pollutant-card .pollutant-value { color: #f1f5f9 !important; }
    .metric-card .sub, .pollutant-card .pollutant-trend { color: #a0afc0 !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { border-color: #29394a; }
    [data-testid="stTabs"] button[data-baseweb="tab"] { color: #a3b2c2; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #75d1c5; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
        background: #152231 !important; border-color: #34465a !important; color: #e6edf5 !important;
    }
    input, textarea { color: #e6edf5 !important; }
    [data-testid="stDataFrame"] { border-color: #2a3a4c; }
    .stButton > button, .stDownloadButton > button { background: #1c3344; color: #edf3f9; border-color: #34495d; }
    </style>
    """, unsafe_allow_html=True)
hero_left.markdown("""
<div class="hero">
  <div class="eyebrow">Air quality intelligence</div>
  <h1>See the air ahead.</h1>
  <p>Explore live atmospheric estimates and an Indian-style AQI forecast, with historical context for 292 places across India.</p>
</div>
""", unsafe_allow_html=True)

tab_live, tab_map, tab_status = st.tabs([
    "🌍 Live Air Quality & Forecast",
    "🗺️ India Map & Compare",
    "ℹ️ Data Sources & Status",
])

# ================================================================== LIVE AIR QUALITY
with tab_live:
    with st.container(border=True):
        section_heading("Choose a location", "Search globally or jump to a place in the India history dataset.", "Explore")
        search_col, india_col, saved_col = st.columns([1.2, 1, 1], gap="medium")
        query = search_col.text_input(
            "Search any city or place",
            placeholder="Try Delhi, Paris, Nairobi…",
            label_visibility="visible",
        )
        pick_options = ["— Select an Indian place —"] + [c for c in HIST_CITIES if place_coords(places_df, c)]
        picked = india_col.selectbox(
            "India history places", pick_options, key="live_pick",
            help="Uses saved coordinates; no location search needed.",
        )
        saved_locations = st.session_state.saved_locations
        saved_labels = [f"{item['name']} · {item['latitude']:.3f}, {item['longitude']:.3f}" for item in saved_locations]
        saved_choice = saved_col.selectbox("Your saved places", ["— None saved —", *saved_labels], key="saved_location_pick")
        saved_location = saved_locations[saved_labels.index(saved_choice)] if saved_choice in saved_labels else None

    if not query and picked == "— Select an Indian place —" and saved_location is None:
        st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
        st.info("Choose a location above to see its current AQI estimate, pollutant outlook, and multi-day forecast.")
    else:
        matches, geo_error = [], None
        if query:
            try:
                matches = cached_geocode(query)
            except LiveDataError as e:
                geo_error = str(e)
        elif picked != "— Select an Indian place —":
            lat0, lon0 = place_coords(places_df, picked)
            st_name = places_df.loc[places_df["City"] == picked, "State"].iloc[0]
            matches = [{"name": picked, "admin1": st_name, "country": "India",
                        "latitude": lat0, "longitude": lon0}]
        else:
            matches = [saved_location]

        if geo_error:
            st.error(geo_error)
        elif not matches:
            st.warning(f"No location found matching '{query}'. Try a different spelling.")
        else:
            labels = [
                f"{m['name']}, {m['admin1'] + ', ' if m['admin1'] else ''}{m['country']}"
                for m in matches
            ]
            if len(labels) > 1:
                choice = st.selectbox("Choose the matching place", labels)
                place = matches[labels.index(choice)]
            else:
                place = matches[0]

            aq_df, aq_error = None, None
            try:
                aq_df, fetched_at = cached_air_quality(place["latitude"], place["longitude"])
            except LiveDataError as e:
                aq_error = str(e)

            if aq_error:
                st.error(aq_error)
            else:
                aq_df = add_predictions(aq_df, reg_model, FEATURES)
                database.save_modelled_frame(place, aq_df)
                record_forecast(place, aq_df)
                actual = aq_df[~aq_df["is_forecast"]]
                forecast = aq_df[aq_df["is_forecast"]]
                current = actual.iloc[-1] if len(actual) else aq_df.iloc[0]

                place_detail = ", ".join(part for part in (place.get("admin1"), place.get("country")) if part)
                section_heading(f"{place['name']} outlook", place_detail or "Live air-quality forecast", "Location")

                forecast_hours = int(forecast["time"].nunique())
                last_forecast_time = forecast["time"].max() if forecast_hours else None
                freshness = fetched_at.astimezone().strftime("%d %b %Y %H:%M %Z")
                st.caption(
                    f"Retrieved {freshness} · {forecast_hours} hourly forecast points available"
                    + (f" through {last_forecast_time.strftime('%d %b %Y %H:%M')} local time." if last_forecast_time is not None else ".")
                )
                if forecast_hours < 24:
                    st.warning("Fewer than 24 forecast hours are available, so the outlook is incomplete.")
                missing_pollutants = [name for name in FEATURES if forecast[name].isna().any()] if forecast_hours else list(FEATURES)
                if missing_pollutants:
                    st.caption("Some pollutant data is missing: " + ", ".join(missing_pollutants) + ". AQI estimates may be less reliable.")

                with st.expander("View location on map"):
                    st.map(pd.DataFrame([{"lat": place["latitude"], "lon": place["longitude"]}]), zoom=8)

                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(
                    metric_card("Current Predicted AQI", f"{current['predicted_aqi']:.0f}"),
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f'<div class="metric-card"><div class="label">Category</div>'
                    f'<div style="margin-top:8px;">{badge_html(current["predicted_bucket"])}</div></div>',
                    unsafe_allow_html=True,
                )
                c3.markdown(metric_card("PM2.5", f"{current['PM2.5']:.0f} µg/m³"), unsafe_allow_html=True)
                c4.markdown(metric_card("PM10", f"{current['PM10']:.0f} µg/m³"), unsafe_allow_html=True)

                st.caption(
                    f"Model-based estimate · latest available timestep {current['time'].strftime('%d %b %Y %H:%M')} local time · "
                    "not a substitute for a nearby regulatory monitor."
                )
                section_heading("AQI outlook", "Recent model estimates followed by the next several days of forecast.", "Forecast")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual["time"], y=actual["predicted_aqi"], mode="lines",
                    name="Recent estimate", line=dict(color="#147b83", width=2.8),
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["time"], y=forecast["predicted_aqi"], mode="lines",
                    name="Forecast", line=dict(color="#e69b45", width=2.8, dash="dash"),
                    fill="tozeroy", fillcolor="rgba(230,155,69,.08)",
                ))
                if len(actual):
                    fig.add_vline(x=actual["time"].iloc[-1], line_dash="dot", line_color="gray")
                fig.update_layout(
                    template="plotly_dark" if dark_mode else "plotly_white", height=370, hovermode="x unified",
                    yaxis_title="Predicted AQI (Indian CPCB scale)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=28, l=8, r=8, b=8),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, linecolor="#e4ebf2"),
                    yaxis=dict(gridcolor="#e9eef4", zeroline=False),
                )
                st.plotly_chart(fig, width='stretch')
                section_heading("Daily forecast", "Average and peak predicted AQI paired with forecast pollutant levels.", "Day by day")
                st.caption(
                    "AQI is model-predicted; pollutant values are Open-Meteo daily averages. "
                    "PM2.5, PM10, NO₂, SO₂ and O₃ are in µg/m³, while CO is in mg/m³."
                )
                daily_forecast = daily_forecast_summary(forecast)
                if daily_forecast.empty:
                    st.info("No future forecast hours are available for this location right now.")
                else:
                    number_columns = [column for column in daily_forecast.columns if column not in {"Date", "Category"}]
                    st.dataframe(
                        daily_forecast.style.format({column: "{:.1f}" for column in number_columns}),
                        width="stretch",
                        hide_index=True,
                    )
                    csv_buffer = StringIO()
                    daily_forecast.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download forecast as CSV", data=csv_buffer.getvalue(),
                        file_name=f"{place['name'].lower().replace(' ', '-')}-aqi-forecast.csv",
                        mime="text/csv",
                    )
                with st.expander("How to read this forecast"):
                    st.caption(
                        "Pollutant concentrations come from Open-Meteo's global air-quality model. AQI values and categories are "
                        "predicted by this project’s model, trained on Indian CPCB station data. They are not Open-Meteo's native "
                        "US/European AQI and are not a substitute for regulatory readings."
                    )

                section_heading("Pollutant snapshot", "Current concentration and the direction of the next 24-hour outlook.", "Pollutants")
                pollutant_row = current[["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]]
                particulate = pollutant_row[["PM2.5", "PM10"]].dropna()
                if not particulate.empty:
                    highest = particulate.idxmax()
                    st.caption(f"Higher particulate concentration: **{highest} · {particulate[highest]:.1f} µg/m³**. Gas values use different units and aren’t ranked against particulates.")
                trends = pollutant_trend_summary(current, forecast)
                if not trends.empty:
                    trend_lookup = trends.set_index("Pollutant")
                    direction_label = {"Rising": "↗ Rising", "Falling": "↘ Falling", "Stable": "→ Stable", "Unavailable": "— No outlook"}
                    for start in range(0, len(trend_lookup), 3):
                        cols = st.columns(3, gap="medium")
                        for col, pollutant in zip(cols, trend_lookup.index[start:start + 3]):
                            row = trend_lookup.loc[pollutant]
                            value = row["Current"]
                            value_label = f"{float(value):.1f}" if pd.notna(value) else "—"
                            average = row["Next 24h avg"]
                            trend = direction_label.get(row["Trend"], row["Trend"])
                            sub = f"{trend}"
                            if pd.notna(average):
                                sub += f" · 24h avg {float(average):.1f} {row['Unit']}"
                            col.markdown(
                                f'<div class="pollutant-card"><div class="pollutant-name">{pollutant}</div>'
                                f'<div class="pollutant-value">{value_label} <span style="font-size:.75rem;font-weight:500;color:#738298">{row["Unit"]}</span></div>'
                                f'<div class="pollutant-trend">{sub}</div></div>',
                                unsafe_allow_html=True,
                            )

                with st.expander("More context, sources & validation"):
                    detail_weather, weather_error = None, None
                    try:
                        detail_weather = cached_weather(place["latitude"], place["longitude"])
                    except LiveDataError as exc:
                        weather_error = str(exc)
                    if detail_weather:
                        st.caption("Weather can influence pollution dispersion but does not prove causation.")
                        w1, w2, w3, w4 = st.columns(4)
                        w1.metric("Temperature", f"{detail_weather['temperature_2m']:.1f} °C")
                        w2.metric("Humidity", f"{detail_weather['relative_humidity_2m']:.0f}%")
                        w3.metric("Wind speed", f"{detail_weather['wind_speed_10m']:.1f} km/h")
                        w4.metric("Rain", f"{detail_weather['precipitation']:.1f} mm")
                    elif weather_error:
                        st.caption(weather_error)

                    st.markdown("**Official monitor**")
                    if official_monitor_configured():
                        try:
                            monitor = cached_official_monitor(place["latitude"], place["longitude"])
                            model_aqi = float(current["predicted_aqi"])
                            monitor_aqi = float(monitor["aqi"])
                            db1, db2, db3 = st.columns(3)
                            db1.metric("Monitor AQI", f"{monitor_aqi:.0f}")
                            db2.metric("Model estimate", f"{model_aqi:.0f}")
                            db3.metric("Absolute difference", f"{abs(monitor_aqi - model_aqi):.0f} AQI")
                            st.caption(
                                f"{monitor['station_name']} · monitor observed {monitor['observed_at']}"
                                + (f" · {monitor['distance_km']} km away" if monitor.get("distance_km") is not None else "")
                            )
                            st.caption("Snapshot comparison only; timestamp and station distance affect how directly these values can be compared.")
                            database.save_official_reading(place, monitor, verified=False, predicted_aqi=model_aqi)
                        except OfficialMonitorError as exc:
                            st.warning(str(exc))
                    else:
                        st.caption("No approved monitor feed is configured. Set `CPCB_AQI_URL` to a licensed endpoint; these model estimates are not official readings.")

                    tracker = forecast_summary(place)
                    st.markdown("**Earlier forecast comparison**")
                    if tracker["samples"]:
                        t1, t2 = st.columns(2)
                        t1.metric("Matured points compared", tracker["samples"])
                        t2.metric("Mean absolute difference", f"{tracker['mae']} AQI points")
                        if tracker.get("by_horizon"):
                            st.dataframe(pd.DataFrame(tracker["by_horizon"]), hide_index=True, width="stretch")
                    else:
                        st.caption("This comparison appears after later collection refreshes some forecast hours.")
                    st.caption("Model-to-model comparison only: an earlier Open-Meteo forecast versus a later Open-Meteo estimate, not official station validation.")

                    if evaluation:
                        st.markdown("**Historical model evaluation**")
                        st.caption("Chronological holdout metrics; they measure historical CPCB daily data, not live Open-Meteo estimates.")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Holdout MAE", evaluation["mae"])
                        m2.metric("Holdout R²", evaluation["r2"])
                        m3.metric("Category accuracy", f"{evaluation['cpcb_category_accuracy']:.1%}")

                    hist_match = place["name"] if place["name"] in HIST_CITIES else None
                    if hist_match is None and place.get("country") == "India":
                        hits = [city for city in HIST_CITIES if city.lower() == place["name"].lower()
                                or city.lower().startswith(place["name"].lower() + " (")]
                        hist_match = hits[0] if len(hits) == 1 else None
                    if hist_match:
                        historical = hist_df[hist_df["City"] == hist_match]
                        st.caption(
                            f"Historical context for {hist_match}: average AQI {historical['AQI'].mean():.0f} across "
                            f"{len(historical):,} days on record; {(historical['AQI'] > 200).mean() * 100:.0f}% of days were Poor or worse."
                        )

                with st.expander("Save this place & set an email alert"):
                    alert_key = f"{place['name']}|{place['latitude']:.5f}|{place['longitude']:.5f}"
                    saved_locations = st.session_state.saved_locations
                    existing_saved = next((item for item in saved_locations
                                           if round(item["latitude"], 5) == round(float(place["latitude"]), 5)
                                           and round(item["longitude"], 5) == round(float(place["longitude"]), 5)), None)
                    if existing_saved:
                        if st.button("Remove from saved places", key=f"remove_saved_{alert_key}"):
                            st.session_state.saved_locations = [item for item in saved_locations if item is not existing_saved]
                            st.rerun()
                    elif st.button("Save this place", key=f"save_place_{alert_key}"):
                        st.session_state.saved_locations.append(dict(place))
                        st.rerun()

                    st.markdown("**In-app alert for this session**")
                    session_threshold = st.slider("Show a warning above forecast AQI", 50, 500,
                                                   int(st.session_state.in_app_alerts.get(alert_key, 200)),
                                                   step=25, key=f"session_threshold_{alert_key}")
                    watching = st.checkbox("Watch this place while this browser session is open",
                                           value=alert_key in st.session_state.in_app_alerts,
                                           key=f"session_watch_{alert_key}")
                    if watching:
                        st.session_state.in_app_alerts[alert_key] = session_threshold
                        peak = int(forecast["predicted_aqi"].max()) if len(forecast) else int(current["predicted_aqi"])
                        if peak >= session_threshold:
                            st.warning(f"Forecast peak is AQI {peak}, above your session threshold of {session_threshold}.")
                    else:
                        st.session_state.in_app_alerts.pop(alert_key, None)

                    st.caption("Optional scheduled email alerts run hourly when the background collector, shared database, and SMTP settings are configured.")
                    st.caption("Email ownership is not verified in this demo; subscribe only an address you control.")
                    with st.form(key=f"subscribe_form_{alert_key}"):
                        alert_email = st.text_input("Email address", key=f"subscribe_email_{alert_key}", placeholder="you@example.com")
                        threshold = st.slider("Send an alert when forecast AQI reaches", 50, 500, 200, step=25, key=f"subscribe_threshold_{alert_key}")
                        subscribe = st.form_submit_button("Subscribe to alerts")
                    if subscribe:
                        try:
                            database.save_alert_subscription(alert_email, place, threshold)
                            if not existing_saved:
                                st.session_state.saved_locations.append(dict(place))
                            st.success("Alert saved. The scheduled collector will email at most once per day when the forecast crosses this threshold.")
                        except ValueError as exc:
                            st.error(str(exc))

                    with st.form(key=f"unsubscribe_form_{alert_key}"):
                        remove_email = st.text_input("Email to remove", key=f"unsubscribe_email_{alert_key}", placeholder="Enter the subscribed address")
                        unsubscribe = st.form_submit_button("Remove this email alert")
                    if unsubscribe:
                        matches_for_email = ([item for item in database.list_alert_subscriptions(remove_email)
                                              if round(item["latitude"], 5) == round(float(place["latitude"]), 5)
                                              and round(item["longitude"], 5) == round(float(place["longitude"]), 5)]
                                             if "@" in remove_email else [])
                        if matches_for_email and database.remove_alert_subscription(matches_for_email[0]["id"], remove_email):
                            st.success("Email alert removed.")
                        else:
                            st.info("No matching alert was found for that email and place.")

# ================================================================== INDIA MAP & COMPARE
with tab_map:
    section_heading("India at a glance", "Explore the latest historical daily AQI and compare places side by side.", "Historical context")
    latest_by_city = hist_df.sort_values("Date").groupby("City", as_index=False).tail(1)
    mapped = latest_by_city.merge(places_df[["City", "latitude", "longitude"]], on="City", how="left").dropna(subset=["latitude", "longitude"])
    map_col, compare_col = st.columns([1.35, 1], gap="large")
    with map_col:
        with st.container(border=True):
            section_heading("Latest readings", "Hover over a marker for the latest available AQI and date.")
            geo = go.Figure(go.Scattergeo(
                lon=mapped["longitude"], lat=mapped["latitude"],
                text=mapped.apply(lambda row: f"{row.City}, {row.State}<br>AQI {row.AQI:.0f}<br>{row.Date.date()}", axis=1),
                hoverinfo="text", mode="markers",
                marker=dict(size=8, color=mapped["AQI"], colorscale="RdYlGn_r", cmin=0, cmax=500,
                            colorbar=dict(title="AQI"), line=dict(width=0.5, color="white")),
            ))
            geo.update_geos(scope="asia", center=dict(lat=22, lon=80), projection_scale=3.8,
                            showland=True, landcolor="#172638" if dark_mode else "#f4f7fa", showcountries=True,
                            showocean=True, oceancolor="#111d2b" if dark_mode else "#f8fbfd", showlakes=True,
                            lakecolor="#111d2b" if dark_mode else "#f8fbfd", countrycolor="#43566b" if dark_mode else "#ffffff")
            geo.update_layout(height=480, margin=dict(t=8, l=0, r=0, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(geo, width="stretch")
            st.caption("Historical daily records only — not live monitor measurements.")

    with compare_col:
        with st.container(border=True):
            section_heading("Compare places", "Select up to three cities to compare their historical records.")
            choices = st.multiselect("Places", HIST_CITIES, default=["Delhi", "Mumbai"] if "Mumbai" in HIST_CITIES else ["Delhi"], max_selections=3, label_visibility="collapsed")
            if choices:
                comparison = (hist_df[hist_df["City"].isin(choices)].groupby("City").agg(
                    state=("State", "first"), days=("AQI", "count"), average_aqi=("AQI", "mean"),
                    poor_or_worse_pct=("AQI", lambda values: (values > 200).mean() * 100), latest_date=("Date", "max")
                ).reset_index().sort_values("average_aqi", ascending=False))
                comparison = comparison.rename(columns={"City": "Place", "state": "State", "days": "Days", "average_aqi": "Avg AQI", "poor_or_worse_pct": "Poor+ days", "latest_date": "Latest data"})
                st.dataframe(comparison.style.format({"Avg AQI": "{:.0f}", "Poor+ days": "{:.0f}%"}), width="stretch", hide_index=True)

# ================================================================== DATA SOURCES & STATUS
with tab_status:
    section_heading("Data, model & storage", "Check source files, collection history, and the limits of the current estimates.", "System status")
    file_status = []
    for label, path in [
        ("Historical CPCB city data", DATA_DIR / "city_day.csv"),
        ("Historical 2022–2025 AQI data", DATA_DIR / "aqi_daily_2022_2025.csv.gz"),
        ("AQI model", MODELS_DIR / "india_aqi_regressor.joblib"),
        ("Evaluation report", OUTPUTS_DIR / "model_evaluation.json"),
    ]:
        file_status.append({"asset": label, "last modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y %H:%M") if path.exists() else "Missing"})
    db_status = database.status()
    s1, s2, s3 = st.columns(3)
    s1.metric("Stored observations", f"{db_status['total_observations']:,}")
    s2.metric("Forecast snapshots", f"{db_status['forecast_snapshots']:,}")
    s3.metric("Verified training rows", f"{db_status['training_eligible']:,}")
    status_left, status_right = st.columns([1.05, .95], gap="large")
    with status_left, st.container(border=True):
        section_heading("Project assets", "Most recent modification time for local data and model files.")
        st.dataframe(pd.DataFrame(file_status), width="stretch", hide_index=True)
    with status_right, st.container(border=True):
        section_heading("Database activity", "Saved records by type.")
        st.json(db_status["by_type"], expanded=False)
        st.caption("Backend: managed PostgreSQL" if DATABASE_URL else f"Backend: local SQLite · `{database.AIR_QUALITY_DB_PATH}`")
    with st.container(border=True):
        section_heading("Forecast validation", "See how model estimates compare with later forecasts and configured monitor readings.", "Model quality")
        monitor_report = database.official_validation_summary()
        if not monitor_report.empty:
            paired_count = int(monitor_report["Paired readings"].sum())
            weighted_mae = float((monitor_report["MAE (AQI)"] * monitor_report["Paired readings"]).sum() / paired_count)
            vm1, vm2 = st.columns(2)
            vm1.metric("Monitor comparisons", f"{paired_count:,}")
            vm2.metric("Monitor comparison MAE", f"{weighted_mae:.1f} AQI")
            st.dataframe(monitor_report, width="stretch", hide_index=True)
            st.caption("These are approximate paired snapshots; station distance and observation timing vary. They are monitoring diagnostics, not a controlled validation study or training labels.")
        else:
            st.caption("Official-monitor validation will appear after a licensed monitor endpoint is configured and readings are collected.")

        forecast_report = forecast_accuracy_by_location()
        st.markdown("**Matured forecast comparison by place and lead time**")
        if forecast_report.empty:
            st.caption("This table fills as saved forecast hours mature and are refreshed by a later model run.")
        else:
            st.dataframe(forecast_report, width="stretch", hide_index=True)
            st.caption("MAE compares an earlier Open-Meteo-based AQI forecast with a later model estimate, not official monitor observations.")
    if db_status["recent_model_runs"]:
        with st.expander("Recent model training runs"):
            st.dataframe(pd.DataFrame(db_status["recent_model_runs"]), width="stretch", hide_index=True)
    with st.expander("Sources and interpretation"):
        st.markdown("""
        - **Live estimates:** Open-Meteo/CAMS atmospheric-model concentrations; global coverage is coarser than a local monitor.
        - **Historical records:** 2015–2020 pollutant/AQI data for 26 cities and daily AQI from Apr 2022–Apr 2025 for 292 places.
        - **Official readings:** only shown after an approved, licensed monitor feed is configured; never silently substituted for a model estimate.
        - **Forecast comparison:** compares a saved forecast with a later Open-Meteo refresh; it is not station validation.
        """)
        if official_monitor_configured():
            st.success("An official-monitor endpoint is configured. Validate its license, station identity and refresh time before relying on it.")
        else:
            st.caption("Official-monitor comparison is unconfigured. Add an approved endpoint through `CPCB_AQI_URL`.")
