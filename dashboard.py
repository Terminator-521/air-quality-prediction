"""
dashboard.py
------------
Streamlit dashboard for the Air Quality Prediction project.

Run with:
    streamlit run dashboard.py

Four tabs:
  1. Live Air Quality & Forecast -- search ANY place on Earth (via
     Open-Meteo's free geocoding API), pull its live + 4-day-forecast
     pollutant readings (via Open-Meteo's free air-quality API), and run
     them through our own Random Forest model (trained on
     Indian CPCB station data) to get a predicted Indian-style AQI value
     and category for right now AND for the next few days. Open-Meteo
     gives raw pollutant forecasts but only computes US/European AQI --
     this is what turns that into an Indian CPCB-style AQI forecast.
  2. Crop Burning Season Impact -- how much the Oct 15-Nov 30 stubble
     burning window pushes up AQI in North Indian cities, from the
     historical dataset.
  3. City History Explorer -- browse the daily AQI record (2015-2020 and
     2022-2025) for any of 292 Indian places, filterable by state.
  4. AQI Assistant -- chat with an LLM (Claude) that answers questions by
     calling this project's own tools: live AQI, forecast, city history,
     city comparison, crop-burning impact and CPCB-based health guidance.
     Needs an Anthropic API key (entered in the sidebar).
"""

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_prep_india import (
    load_history, load_places, place_coords, CROP_BURNING_HOTSPOT_CITIES, search_city,
)
from live_data import geocode, fetch_air_quality, add_predictions, LiveDataError
from assistant_tools import AirQualityTools
from assistant import (
    MODELS, DEFAULT_MODEL_LABEL, SAMPLE_QUESTIONS,
    get_api_key, make_client, run_assistant, friendly_error,
)

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
    return load_history()

@st.cache_data
def load_place_table():
    return load_places()


def with_gaps(df, days=45):
    """Insert a blank row wherever the data jumps by more than `days`, so charts
    show the Jul 2020 - Mar 2022 gap as a break instead of a misleading line."""
    df = df.sort_values("Date").reset_index(drop=True)
    gaps = df.index[df["Date"].diff().dt.days > days]
    if len(gaps) == 0:
        return df
    blanks = pd.DataFrame({"Date": df.loc[gaps, "Date"] - pd.Timedelta(days=1), "AQI": float("nan")})
    return pd.concat([df, blanks], ignore_index=True).sort_values("Date")


def shade_seasons(fig, dates):
    for year in sorted(dates.dt.year.unique()):
        fig.add_vrect(
            x0=pd.Timestamp(year=year, month=10, day=15), x1=pd.Timestamp(year=year, month=11, day=30),
            fillcolor="#E53935", opacity=0.15, line_width=0,
        )

@st.cache_data(ttl=900)
def cached_geocode(query):
    return geocode(query)

@st.cache_data(ttl=900)
def cached_air_quality(lat, lon):
    return fetch_air_quality(lat, lon)


clf_model, reg_model, FEATURES = load_models()
hist_df = load_historical()
places_df = load_place_table()
HIST_CITIES = sorted(hist_df["City"].unique())
HIST_STATES = sorted(hist_df["State"].dropna().unique())


@st.cache_resource
def load_assistant_tools():
    return AirQualityTools(clf_model, reg_model, FEATURES, hist_df)


assistant_tools = load_assistant_tools()
st.session_state.setdefault("chat_history", [])   # plain text turns sent to the LLM
st.session_state.setdefault("chat_display", [])   # what the chat UI shows (+ tool traces)

# ---------------------------------------------------------------- sidebar: assistant settings
with st.sidebar:
    st.header("🤖 Assistant settings")
    key_input = st.text_input(
        "Anthropic API key", type="password", placeholder="sk-ant-...",
        help="Only used by the AQI Assistant tab. Kept in this browser session; never saved to disk.",
    )
    model_label = st.selectbox(
        "Model", list(MODELS), index=list(MODELS).index(DEFAULT_MODEL_LABEL),
    )
    api_key = get_api_key(key_input)
    if api_key and not key_input:
        st.caption("Using the key from ANTHROPIC_API_KEY / Streamlit secrets.")
    st.caption("Get a key at console.anthropic.com. The other three tabs need no key.")

st.markdown("""
<div class="hero">
  <h1>🌍 Air Quality Prediction &amp; Monitoring</h1>
  <p>Live global air quality (Open-Meteo — free, no API key) scored by an ML model trained on
     Indian CPCB station data, with daily AQI history for 292 places across India (2015–2025)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Live Air Quality & Forecast",
    "🔥 Crop Burning Season Impact",
    "📈 City History Explorer",
    "🤖 AQI Assistant",
])

# ================================================================== TAB 1
with tab1:
    st.subheader("Search any location, anywhere")
    query = st.text_input(
        "City, town, or place name",
        placeholder="e.g. Delhi, Paris, New York, Nairobi, Tokyo",
    )

    pick_options = ["—"] + [c for c in HIST_CITIES if place_coords(places_df, c)]
    picked = st.selectbox(
        "…or pick one of the 292 Indian places we have history for", pick_options, key="live_pick",
        help="Uses saved coordinates, so no location search is needed.",
    )

    if not query and picked == "—":
        st.info("Type a place name above (or pick an Indian place) to fetch its live air quality and a 4-day forecast.")
    else:
        matches, geo_error = [], None
        if query:
            try:
                matches = cached_geocode(query)
            except LiveDataError as e:
                geo_error = str(e)
        else:
            lat0, lon0 = place_coords(places_df, picked)
            st_name = places_df.loc[places_df["City"] == picked, "State"].iloc[0]
            matches = [{"name": picked, "admin1": st_name, "country": "India",
                        "latitude": lat0, "longitude": lon0}]

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

                hist_match = place["name"] if place["name"] in HIST_CITIES else None
                if hist_match is None and place["country"] == "India":
                    hits = [c for c in HIST_CITIES if c.lower() == place["name"].lower()
                            or c.lower().startswith(place["name"].lower() + " (")]
                    hist_match = hits[0] if len(hits) == 1 else None
                if hist_match:
                    hd = hist_df[hist_df["City"] == hist_match]
                    st.caption(
                        f"📚 Historical context for {hist_match}: average AQI **{hd['AQI'].mean():.0f}** over "
                        f"{len(hd):,} days on record; **{(hd['AQI'] > 200).mean() * 100:.0f}%** of days were "
                        f"Poor or worse. See the History Explorer tab for more."
                    )

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
                    "prediction, trained on Indian CPCB station data — not Open-Meteo's native "
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
    st.caption(
        "Data: daily AQI from 2015-2020 (26 cities) and 2022-2025 (292 places). "
        "Season windows covered: 2015-2019 (few cities in the early years) and 2022-2024 (most places)."
    )

    hotspot_present = [c for c in CROP_BURNING_HOTSPOT_CITIES if c in HIST_CITIES]
    season_df = hist_df.copy()
    season_df["group"] = season_df["City"].apply(
        lambda c: c if c in hotspot_present else "Other places"
    )
    summary = (
        season_df.groupby(["group", "is_crop_burning_season"])["AQI"]
        .mean()
        .unstack()
        .rename(columns={False: "Rest of year", True: "Crop-burning season"})
    )
    summary["Increase"] = summary["Crop-burning season"] - summary["Rest of year"]
    summary = summary.loc[hotspot_present + ["Other places"]]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(summary.style.format("{:.0f}"), width='stretch')
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Rest of year", x=summary.index, y=summary["Rest of year"], marker_color="#2a5298"))
        fig.add_trace(go.Bar(name="Crop-burning season", x=summary.index, y=summary["Crop-burning season"], marker_color="#E53935"))
        fig.update_layout(barmode="group", template="plotly_white", height=350, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Season by season: average AQI in the burning window")
    season_years = sorted(hist_df.loc[hist_df["is_crop_burning_season"], "year"].unique())
    yr = (
        hist_df[hist_df["is_crop_burning_season"] & hist_df["City"].isin(hotspot_present)]
        .groupby(["City", "year"])["AQI"].mean().unstack().reindex(hotspot_present)
    )
    fig_y = go.Figure()
    for y in season_years:
        if y in yr.columns:
            fig_y.add_trace(go.Bar(name=str(y), x=yr.index, y=yr[y]))
    fig_y.update_layout(barmode="group", template="plotly_white", height=330,
                        yaxis_title="Avg AQI, 15 Oct - 30 Nov", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_y, width='stretch')

    st.subheader("Which states are hit hardest?")
    by_state = (
        hist_df[hist_df["year"] >= 2022].groupby(["State", "is_crop_burning_season"])["AQI"].mean().unstack()
        .rename(columns={False: "Rest of year", True: "Crop-burning season"}).dropna()
    )
    by_state = by_state.sort_values("Crop-burning season", ascending=False).head(12)
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(name="Rest of year", x=by_state.index, y=by_state["Rest of year"], marker_color="#2a5298"))
    fig_s.add_trace(go.Bar(name="Crop-burning season", x=by_state.index, y=by_state["Crop-burning season"], marker_color="#E53935"))
    fig_s.update_layout(barmode="group", template="plotly_white", height=360,
                        margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_s, width='stretch')
    st.caption("2022-2025 data only, averaged over all places in each state (top 12 states by season AQI).")

    st.subheader("Delhi: daily AQI history, crop-burning windows shaded")
    delhi = with_gaps(hist_df[hist_df["City"] == "Delhi"][["Date", "AQI"]])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=delhi["Date"], y=delhi["AQI"], mode="lines", line=dict(color="#2a5298", width=1)))
    shade_seasons(fig2, delhi["Date"])
    fig2.update_layout(template="plotly_white", height=380, yaxis_title="AQI", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig2, width='stretch', key='delhi_chart')
    st.caption("The break between mid-2020 and early 2022 is real: neither dataset covers it.")

    st.subheader("Top 10 most-affected places during crop-burning season")
    seas = hist_df[hist_df["is_crop_burning_season"]]
    n_days = seas.groupby("City")["AQI"].count()
    seas = seas[seas["City"].isin(n_days[n_days >= 30].index)]
    season_avg = seas.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    fig3 = go.Figure(go.Bar(x=season_avg.values, y=season_avg.index, orientation="h", marker_color="#FB8C00"))
    fig3.update_layout(template="plotly_white", height=380, xaxis_title="Average AQI (mid-Oct to end-Nov)",
                        margin=dict(t=10, l=10, r=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, width='stretch')
    st.caption("Places with fewer than 30 in-season days on record are left out.")

# ================================================================== TAB 3
with tab3:
    st.subheader(f"Browse daily AQI history for {len(HIST_CITIES)} Indian places (2015-2025)")
    f1, f2 = st.columns([1, 2])
    state_sel = f1.selectbox("State / UT", ["All states"] + HIST_STATES, key="hist_state")
    hquery = f2.text_input("Type a place name", "", key="hist_search")
    pool = (HIST_CITIES if state_sel == "All states" else
            sorted(hist_df.loc[hist_df["State"] == state_sel, "City"].unique()))
    hmatches = search_city(hquery, pool) if hquery else pool

    if not hmatches:
        st.warning(f"No place matching '{hquery}'" + ("" if state_sel == "All states" else f" in {state_sel}") + ".")
    else:
        default_idx = hmatches.index("Delhi") if "Delhi" in hmatches else 0
        city = st.selectbox(f"Place ({len(hmatches)} available)", hmatches, index=default_idx, key="hist_city")
        city_df = hist_df[hist_df["City"] == city].sort_values("Date")

        coords = place_coords(places_df, city)
        if coords:
            st.map(pd.DataFrame([{"lat": coords[0], "lon": coords[1]}]), zoom=9)
        else:
            st.caption("No map coordinate is available for this place.")

        latest = city_df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Latest AQI on record", f"{latest['AQI']:.0f}", latest["AQI_Bucket"]), unsafe_allow_html=True)
        c2.markdown(metric_card("Average AQI (all years)", f"{city_df['AQI'].mean():.0f}"), unsafe_allow_html=True)
        in_season = city_df[city_df["is_crop_burning_season"]]["AQI"].mean()
        out_season = city_df[~city_df["is_crop_burning_season"]]["AQI"].mean()
        if pd.notna(in_season) and pd.notna(out_season):
            c3.markdown(
                metric_card("Crop-burning season AQI", f"{in_season:.0f}", f"{in_season - out_season:+.0f} vs rest of year"),
                unsafe_allow_html=True,
            )
        else:
            c3.markdown(metric_card("Crop-burning season AQI", "n/a", "no in-season days on record"), unsafe_allow_html=True)
        st.caption(
            f"{city_df['State'].iloc[0]} · {len(city_df):,} days on record "
            f"({city_df['Date'].min().date()} to {city_df['Date'].max().date()})"
        )

        st.write("")
        st.subheader(f"{city}: daily AQI history (crop-burning windows shaded)")
        plot_df = with_gaps(city_df[["Date", "AQI"]])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["AQI"], mode="lines", line=dict(color="#2a5298", width=1)))
        shade_seasons(fig4, city_df["Date"])
        fig4.update_layout(template="plotly_white", height=380, yaxis_title="AQI", margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig4, width='stretch', key='hist_chart')

        dom = city_df["dominant_pollutants"].dropna()
        if len(dom):
            st.subheader("Which pollutant drives the AQI most often?")
            dcount = dom.value_counts().head(6)
            fig5 = go.Figure(go.Bar(x=dcount.index, y=dcount.values / len(dom) * 100, marker_color="#6A1B9A"))
            fig5.update_layout(template="plotly_white", height=280, yaxis_title="% of days (2022-2025)",
                               margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig5, width='stretch', key='dom_chart')
            st.caption("The pollutant with the highest sub-index each day; only the 2022-2025 file lists it.")

        st.subheader("Raw data sample")
        st.dataframe(
            city_df[["Date", "City", "State", "AQI", "AQI_Bucket", "dominant_pollutants", "stations", "source"]].tail(200),
            width='stretch',
        )

# ================================================================== TAB 4
def render_trace(trace):
    """Show which tools the assistant called (and what they returned)."""
    n = len(trace)
    with st.expander(f"🔧 {n} tool call{'s' if n != 1 else ''} used"):
        for t in trace:
            args = ", ".join(f"{k}={v!r}" for k, v in t["input"].items())
            st.markdown(f"**{t['tool']}**({args})")
            st.json(t["output"], expanded=False)


with tab4:
    st.subheader("🤖 AQI Assistant")
    st.caption(
        "Ask in plain English. This is an LLM that answers by calling the project's own tools "
        "(live AQI, forecast, city history, crop-burning impact, health guidance), so its numbers "
        "come from real data, not from memory. Expand '🔧 tool calls' under any answer to see exactly what it used."
    )

    if not api_key:
        st.info("Add your Anthropic API key in the sidebar to start chatting. The other tabs work without one.")

    st.write("**Try asking:**")
    sample_cols = st.columns(2)
    clicked = None
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if sample_cols[i % 2].button(q, key=f"sample_{i}"):
            clicked = q

    if st.session_state.chat_display and st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.session_state.chat_display = []
        st.rerun()

    for m in st.session_state.chat_display:
        with st.chat_message(m["role"]):
            level = m.get("level")
            if level == "error":
                st.error(m["text"])
            elif level == "warning":
                st.warning(m["text"])
            else:
                st.markdown(m["text"])
            if m.get("trace"):
                render_trace(m["trace"])

    prompt = st.chat_input("Ask about air quality…", key="aqi_chat_input") or clicked
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_display.append({"role": "user", "text": prompt})

        with st.chat_message("assistant"):
            if not api_key:
                st.session_state.chat_display.append({
                    "role": "assistant", "level": "warning",
                    "text": "I need an Anthropic API key first. Add it in the sidebar and ask again.",
                })
            else:
                result, err = None, None
                with st.spinner("Checking the data…"):
                    try:
                        result = run_assistant(
                            make_client(api_key), MODELS[model_label],
                            st.session_state.chat_history, prompt, assistant_tools,
                        )
                    except Exception as e:  # SDK errors -> readable message
                        err = friendly_error(e)

                if err:
                    st.session_state.chat_display.append(
                        {"role": "assistant", "level": "error", "text": err})
                else:
                    st.session_state.chat_display.append(
                        {"role": "assistant", "text": result.text, "trace": result.trace})
                    st.session_state.chat_history += [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result.text},
                    ]
                    st.session_state.chat_history = st.session_state.chat_history[-12:]

        # Redraw the tab from saved state so the history (and the Clear chat
        # button, which is drawn above it) reflects this turn immediately.
        st.rerun()
