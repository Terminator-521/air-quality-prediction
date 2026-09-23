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
     them through our own Random Forest model (trained on 5 years of
     Indian CPCB station data) to get a predicted Indian-style AQI value
     and category for right now AND for the next few days. Open-Meteo
     gives raw pollutant forecasts but only computes US/European AQI --
     this is what turns that into an Indian CPCB-style AQI forecast.
  2. Crop Burning Season Impact -- how much the Oct 15-Nov 30 stubble
     burning window pushes up AQI in North Indian cities, from the
     historical dataset.
  3. City History Explorer -- browse the historical (2015-2020) daily
     AQI record for any of the 26 Indian cities in the training data.
"""

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_prep_india import load_and_clean, CITY_COORDS, CROP_BURNING_HOTSPOT_CITIES, search_city
from live_data import geocode, fetch_air_quality, add_predictions, LiveDataError

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="wide")

# ---------------------------------------------------------------- styling
BUCKET_COLORS = {
    "Good": "#2E7D32", "Satisfactory": "#8BC34A", "Moderate": "#F9A825",
    "Poor": "#FB8C00", "Very Poor": "#E53935", "Severe": "#7E0023",
}
BUCKET_TEXT = {
    "Good": "white", "Satisfactory": "#1a1a1a", "Moderate": "#1a1a1a",
    "Poor": "white", "Very Poor": "white", "Severe": "white",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 1.8rem 2.2rem; border-radius: 16px; color: white; margin-bottom: 1.4rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 0.98rem; }

.metric-card {
    background: rgba(127,127,127,0.06); border-radius: 14px; padding: 1rem 1.2rem;
    border-left: 6px solid #2a5298; height: 100%;
}
.metric-card .label { font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.metric-card .sub { font-size: 0.76rem; opacity: 0.6; margin-top: 2px; }

.aqi-badge {
    display: inline-block; padding: 7px 16px; border-radius: 999px;
    font-weight: 700; font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'


def badge_html(bucket, aqi=None):
    bg = BUCKET_COLORS.get(bucket, "#999")
    fg = BUCKET_TEXT.get(bucket, "white")
    val = f" · AQI {aqi:.0f}" if aqi is not None else ""
    return f'<span class="aqi-badge" style="background:{bg};color:{fg};">{bucket}{val}</span>'


# ---------------------------------------------------------------- cached loaders
@st.cache_resource
def load_models():
    clf = joblib.load("models/india_aqi_classifier.joblib")
    reg = joblib.load("models/india_aqi_regressor.joblib")
    features = joblib.load("models/india_features.joblib")
    return clf, reg, features

@st.cache_data
def load_historical():
    return load_and_clean()

@st.cache_data(ttl=900)
def cached_geocode(query):
    return geocode(query)

@st.cache_data(ttl=900)
def cached_air_quality(lat, lon):
    return fetch_air_quality(lat, lon)


clf_model, reg_model, FEATURES = load_models()
hist_df = load_historical()
HIST_CITIES = sorted(hist_df["City"].unique())

st.markdown("""
<div class="hero">
  <h1>🌍 Air Quality Prediction &amp; Monitoring</h1>
  <p>Live global air quality (Open-Meteo — free, no API key) scored by an ML model trained on
     5 years of Indian CPCB station data</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🌍 Live Air Quality & Forecast",
    "🔥 Crop Burning Season Impact",
    "📈 City History Explorer",
])

# ================================================================== TAB 1
with tab1:
    st.subheader("Search any location, anywhere")
    query = st.text_input(
        "City, town, or place name",
        placeholder="e.g. Delhi, Paris, New York, Nairobi, Tokyo",
    )

    if not query:
        st.info("Type a place name above to fetch its live air quality and a 4-day forecast.")
    else:
        matches, geo_error = [], None
        try:
            matches = cached_geocode(query)
        except LiveDataError as e:
            geo_error = str(e)

        if geo_error:
            st.error(geo_error)
        elif not matches:
            st.warning(f"No location found matching '{query}'. Try a different spelling.")
        else:
            labels = [
                f"{m['name']}, {m['admin1'] + ', ' if m['admin1'] else ''}{m['country']}"
                for m in matches
            ]
            choice = st.selectbox("Matching locations", labels)
            place = matches[labels.index(choice)]

            aq_df, aq_error = None, None
            try:
                aq_df = cached_air_quality(place["latitude"], place["longitude"])
            except LiveDataError as e:
                aq_error = str(e)

            if aq_error:
                st.error(aq_error)
            else:
                aq_df = add_predictions(aq_df, clf_model, reg_model, FEATURES)
                actual = aq_df[~aq_df["is_forecast"]]
                forecast = aq_df[aq_df["is_forecast"]]
                current = actual.iloc[-1] if len(actual) else aq_df.iloc[0]

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

                st.write("")
                n_forecast_days = forecast["time"].dt.date.nunique() if len(forecast) else 0
                st.subheader(f"{place['name']}: recent actuals + {n_forecast_days}-day forecast")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=actual["time"], y=actual["predicted_aqi"], mode="lines",
                    name="Actual (recent)", line=dict(color="#2a5298", width=2.5),
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["time"], y=forecast["predicted_aqi"], mode="lines",
                    name="Forecast", line=dict(color="#e53935", width=2.5, dash="dash"),
                ))
                if len(actual):
                    fig.add_vline(x=actual["time"].iloc[-1], line_dash="dot", line_color="gray")
                fig.update_layout(
                    template="plotly_white", height=420, hovermode="x unified",
                    yaxis_title="Predicted AQI (Indian CPCB scale)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=30, l=10, r=10, b=10),
                )
                st.plotly_chart(fig, width='stretch')
                st.caption(
                    "Pollutant concentrations (PM2.5, PM10, CO, NO2, SO2, O3) come live from Open-Meteo's "
                    "global air-quality model. The AQI value and category shown are **our own** model's "
                    "prediction, trained on 5 years of Indian CPCB station data — not Open-Meteo's native "
                    "AQI, which is computed on the US/European scale."
                )

                st.subheader("Current pollutant breakdown")
                pollutant_row = current[["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]]
                bar = go.Figure(go.Bar(
                    x=list(pollutant_row.index), y=list(pollutant_row.values),
                    marker_color="#2a5298",
                ))
                bar.update_layout(
                    template="plotly_white", height=300,
                    yaxis_title="µg/m³ (CO shown in mg/m³)",
                    margin=dict(t=10, l=10, r=10, b=10),
                )
                st.plotly_chart(bar, width='stretch')

# ================================================================== TAB 2
with tab2:
    st.subheader("What is 'crop burning season'?")
    st.write(
        "Every year, roughly **mid-October through the end of November**, farmers "
        "across Punjab and Haryana burn leftover paddy stubble to quickly clear "
        "fields before the next planting cycle. The smoke plume drifts across "
        "North India, and is one of the biggest documented drivers of the annual "
        "air-quality crisis in Delhi-NCR and nearby cities."
    )

    hotspot_present = [c for c in CROP_BURNING_HOTSPOT_CITIES if c in HIST_CITIES]
    season_df = hist_df.copy()
    season_df["group"] = season_df["City"].apply(
        lambda c: c if c in hotspot_present else "Other cities"
    )
    summary = (
        season_df.groupby(["group", "is_crop_burning_season"])["AQI"]
        .mean()
        .unstack()
        .rename(columns={False: "Rest of year", True: "Crop-burning season"})
    )
    summary["Increase"] = summary["Crop-burning season"] - summary["Rest of year"]
    summary = summary.loc[hotspot_present + ["Other cities"]]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(summary.style.format("{:.0f}"), width='stretch')
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Rest of year", x=summary.index, y=summary["Rest of year"], marker_color="#2a5298"))
        fig.add_trace(go.Bar(name="Crop-burning season", x=summary.index, y=summary["Crop-burning season"], marker_color="#E53935"))
        fig.update_layout(barmode="group", template="plotly_white", height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Delhi: full daily AQI history, crop-burning windows shaded")
    delhi = hist_df[hist_df["City"] == "Delhi"].sort_values("Date")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=delhi["Date"], y=delhi["AQI"], mode="lines", line=dict(color="#2a5298", width=1)))
    for year in delhi["Date"].dt.year.unique():
        fig2.add_vrect(
            x0=pd.Timestamp(year=year, month=10, day=15), x1=pd.Timestamp(year=year, month=11, day=30),
            fillcolor="#E53935", opacity=0.15, line_width=0,
        )
    fig2.update_layout(template="plotly_white", height=380, yaxis_title="AQI", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig2, width='stretch')

    st.subheader("Top 10 most-affected cities during crop-burning season")
    season_avg = (
        hist_df[hist_df["is_crop_burning_season"]]
        .groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    )
    fig3 = go.Figure(go.Bar(x=season_avg.values, y=season_avg.index, orientation="h", marker_color="#FB8C00"))
    fig3.update_layout(template="plotly_white", height=380, xaxis_title="Average AQI (mid-Oct to end-Nov)",
                        margin=dict(t=10, l=10, r=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, width='stretch')

# ================================================================== TAB 3
with tab3:
    st.subheader("Browse historical data (2015-2020) for any of the 26 tracked cities")
    hquery = st.text_input("Type a city name", "", key="hist_search")
    hmatches = search_city(hquery, HIST_CITIES) if hquery else HIST_CITIES

    if not hmatches:
        st.warning(f"No city matching '{hquery}'.")
    else:
        city = st.selectbox("City", hmatches, key="hist_city")
        city_df = hist_df[hist_df["City"] == city].sort_values("Date")

        if city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]
            st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=9)

        latest = city_df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Latest AQI on record", f"{latest['AQI']:.0f}", latest["AQI_Bucket"]), unsafe_allow_html=True)
        c2.markdown(metric_card("Average AQI (all years)", f"{city_df['AQI'].mean():.0f}"), unsafe_allow_html=True)
        in_season = city_df[city_df["is_crop_burning_season"]]["AQI"].mean()
        out_season = city_df[~city_df["is_crop_burning_season"]]["AQI"].mean()
        c3.markdown(
            metric_card("Crop-burning season AQI", f"{in_season:.0f}", f"{in_season - out_season:+.0f} vs rest of year"),
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader(f"{city}: daily AQI history (crop-burning windows shaded)")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=city_df["Date"], y=city_df["AQI"], mode="lines", line=dict(color="#2a5298", width=1)))
        for year in city_df["Date"].dt.year.unique():
            fig4.add_vrect(
                x0=pd.Timestamp(year=year, month=10, day=15), x1=pd.Timestamp(year=year, month=11, day=30),
                fillcolor="#E53935", opacity=0.15, line_width=0,
            )
        fig4.update_layout(template="plotly_white", height=380, yaxis_title="AQI", margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig4, width='stretch')

        st.subheader("Raw data sample")
        st.dataframe(city_df.tail(200), width='stretch')
