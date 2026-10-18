"""
assistant_tools.py
------------------
The "tools" the AQI chat assistant (an LLM) can call.

Each tool is a plain Python method that reuses code this project already
has -- live Open-Meteo data (live_data.py), the trained Random Forest
models (models/*.joblib) and the merged CPCB daily-AQI history (data_prep_india.py).
The LLM never invents air quality numbers: it has to call a tool, and the
tool returns real data (or a clear error).

Flow:
    user question -> LLM decides which tool(s) to call -> this module runs
    them -> results go back to the LLM -> LLM writes the final answer.

Every tool returns a JSON-serialisable dict. Errors are returned as
{"error": "..."} instead of raised, so the LLM can explain the problem to
the user instead of the app crashing.
"""

import time

import pandas as pd

import live_data
from data_prep_india import search_city

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
PROFILES = [
    "general", "asthma_or_lung_disease", "heart_disease", "child",
    "elderly", "pregnant", "runner_or_athlete", "outdoor_worker",
]

# Simplified plain-language summaries in the spirit of the CPCB National AQI health
# categories (not a verbatim quote -- check the official CPCB document before citing).
CATEGORY_IMPACT = {
    "Good": "Minimal health impact.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": ("May cause breathing discomfort to people with lung disease such as "
                 "asthma, and discomfort to people with heart disease, children and "
                 "older adults."),
    "Poor": "May cause breathing discomfort to most people on prolonged exposure.",
    "Very Poor": "May cause respiratory illness on prolonged exposure.",
    "Severe": ("May affect healthy people and seriously affect those with existing "
               "lung or heart disease, even during light physical activity."),
}

# General precautions (common public-health guidance, not CPCB wording).
CATEGORY_PRECAUTIONS = {
    "Good": "No special precautions needed.",
    "Satisfactory": "Unusually sensitive people may prefer to go easy on long outdoor exertion.",
    "Moderate": ("Sensitive groups should reduce prolonged or heavy outdoor exertion "
                 "and keep any prescribed quick-relief medication handy."),
    "Poor": ("Limit prolonged outdoor exertion for everyone; sensitive groups should stay "
             "indoors where possible. A well-fitted N95/FFP2 mask helps outdoors."),
    "Very Poor": ("Avoid outdoor exercise, keep windows closed, use an air purifier if "
                  "available, and wear a well-fitted N95 if you must go out."),
    "Severe": ("Stay indoors and avoid outdoor exertion. Use an air purifier if available "
               "and wear a well-fitted N95 for essential travel. Seek medical care for "
               "breathing difficulty or chest pain."),
}

PROFILE_EXTRAS = {
    "general": "",
    "asthma_or_lung_disease": ("Carry your prescribed reliever inhaler and follow your "
                               "doctor's action plan; avoid outdoor exertion from "
                               "Moderate upward."),
    "heart_disease": ("Avoid strenuous outdoor activity from Moderate upward; get "
                      "medical help for chest pain, unusual breathlessness or palpitations."),
    "child": ("Children breathe faster and are more affected: limit outdoor play from "
              "Moderate upward and prefer indoor activities on Poor or worse days."),
    "elderly": ("Older adults are more sensitive: avoid long outdoor walks from Moderate "
                "upward and prefer indoor activity on Poor or worse days."),
    "pregnant": ("Cut time outdoors on Poor or worse days, and talk to your doctor if "
                 "you are concerned."),
    "runner_or_athlete": ("Exercise increases how much polluted air you breathe in: move "
                          "workouts indoors from Poor upward, and on Moderate days prefer "
                          "lighter sessions at the cleanest hour of the day."),
    "outdoor_worker": ("Use a well-fitted N95 at Poor and worse levels and take breaks in "
                       "cleaner indoor air where you can."),
}

DISCLAIMER = ("General information based on the estimated AQI category, not medical advice. "
              "People with health conditions should follow their doctor's guidance.")

HISTORY_NOTE = ("Historical daily AQI: 26 cities from 2015 to mid-2020 plus 292 places "
                "(32 states/UTs) from Apr 2022 to Apr 2025, with no data for mid-2020 to "
                "Mar 2022. It shows typical patterns, not current conditions.")

# Common alternative spellings -> name used in the dataset
ALIASES = {
    "gurgaon": "Gurugram", "bangalore": "Bengaluru", "bombay": "Mumbai",
    "calcutta": "Kolkata", "madras": "Chennai", "trivandrum": "Thiruvananthapuram",
    "vizag": "Visakhapatnam", "new delhi": "Delhi", "cochin": "Kochi",
    "mysore": "Mysuru", "banaras": "Varanasi", "benares": "Varanasi",
    "allahabad": "Prayagraj", "baroda": "Vadodara", "poona": "Pune",
}

MIN_DAYS_RANKING = 365      # places with less data are left out of rankings
MIN_SEASON_DAYS = 30        # ...and out of crop-season rankings
METRICS = ["avg_aqi", "crop_season_aqi", "pct_poor_or_worse_days"]


def _r(x, n=1):
    """Round to a plain Python float (keeps results JSON-serialisable)."""
    return float(round(float(x), n))


class AirQualityTools:
    """Holds the models + data and exposes the tools the LLM can call."""

    def __init__(self, clf, reg, features, hist_df, geocode_fn=None, fetch_fn=None, ttl=900):
        self.clf = clf
        self.reg = reg
        self.features = features
        self.hist = hist_df
        self._geocode_fn = geocode_fn
        self._fetch_fn = fetch_fn
        self._ttl = ttl
        self._cache = {}
        self.cities = sorted(hist_df["City"].unique())
        self.states = sorted(hist_df["State"].dropna().unique())

        self._dispatch = {
            "get_current_aqi": self.get_current_aqi,
            "get_aqi_forecast": self.get_aqi_forecast,
            "get_city_history": self.get_city_history,
            "compare_cities": self.compare_cities,
            "get_crop_burning_impact": self.get_crop_burning_impact,
            "get_health_guidance": self.get_health_guidance,
            "rank_places": self.rank_places,
        }

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def call(self, name, args):
        """Run a tool by name. Never raises: failures come back as {"error": ...}."""
        fn = self._dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**(args or {}))
        except live_data.LiveDataError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:  # keep the chat alive whatever happens
            return {"error": f"{name} failed unexpectedly ({type(e).__name__})."}

    # ------------------------------------------------------------------
    # live tools
    # ------------------------------------------------------------------
    def _live_frame(self, place):
        """Geocode a place, fetch live+forecast data, run the models. Cached ~15 min."""
        geocode = self._geocode_fn or live_data.geocode
        fetch = self._fetch_fn or live_data.fetch_air_quality

        matches = geocode(place, count=3)
        if not matches:
            raise live_data.LiveDataError(
                f"No location found matching '{place}'. Try a different spelling.")
        loc = matches[0]
        key = (round(loc["latitude"], 2), round(loc["longitude"], 2))

        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return loc, matches, hit[1]

        df = fetch(loc["latitude"], loc["longitude"])
        df = live_data.add_predictions(df, self.clf, self.reg, self.features)
        self._cache[key] = (time.monotonic(), df)
        return loc, matches, df

    @staticmethod
    def _label(loc):
        parts = [loc["name"], loc.get("admin1", ""), loc.get("country", "")]
        return ", ".join(p for p in parts if p)

    def get_current_aqi(self, place):
        loc, matches, df = self._live_frame(place)
        actual = df[~df["is_forecast"]]
        row = actual.iloc[-1] if len(actual) else df.iloc[0]
        out = {
            "location": self._label(loc),
            "as_of_local_time": row["time"].strftime("%Y-%m-%d %H:%M"),
            "estimated_aqi": _r(row["predicted_aqi"], 0),
            "category": str(row["predicted_bucket"]),
            "pollutants_ug_m3": {
                "PM2.5": _r(row["PM2.5"]), "PM10": _r(row["PM10"]),
                "NO2": _r(row["NO2"]), "SO2": _r(row["SO2"]), "O3": _r(row["O3"]),
            },
            "CO_mg_m3": _r(row["CO"], 2),
            "note": ("AQI is estimated on the Indian CPCB scale by this project's model "
                     "from Open-Meteo pollutant data."),
        }
        if len(matches) > 1:
            out["other_possible_matches"] = [self._label(m) for m in matches[1:]]
        return out

    def get_aqi_forecast(self, place, days=3):
        days = max(1, min(int(days), 4))
        loc, _, df = self._live_frame(place)
        fc = df[df["is_forecast"]]
        if fc.empty:
            raise live_data.LiveDataError("No forecast data is available for this location.")

        daily = []
        for date, g in list(fc.groupby(fc["time"].dt.date))[:days]:
            worst = g.loc[g["predicted_aqi"].idxmax()]
            best = g.loc[g["predicted_aqi"].idxmin()]
            daily.append({
                "date": str(date),
                "hours_covered": int(len(g)),
                "avg_aqi": _r(g["predicted_aqi"].mean(), 0),
                "peak_aqi": _r(worst["predicted_aqi"], 0),
                "peak_category": str(worst["predicted_bucket"]),
                "worst_hour": worst["time"].strftime("%H:00"),
                "cleanest_hour": best["time"].strftime("%H:00"),
                "cleanest_hour_aqi": _r(best["predicted_aqi"], 0),
            })

        # Trend: next 24 forecast hours vs the last 24 actual hours
        actual = df[~df["is_forecast"]]
        trend = None
        if len(actual) >= 12:
            recent = actual["predicted_aqi"].tail(24).mean()
            nxt = fc["predicted_aqi"].head(24).mean()
            change = (nxt - recent) / max(recent, 1.0)
            trend = {
                "last_24h_avg_aqi": _r(recent, 0),
                "next_24h_avg_aqi": _r(nxt, 0),
                "direction": ("worsening" if change > 0.10 else
                              "improving" if change < -0.10 else "roughly steady"),
            }

        return {
            "location": self._label(loc),
            "daily_forecast": daily,
            "trend": trend,
            "note": ("Estimated AQI from Open-Meteo's pollutant forecast run through this "
                     "project's model. Treat later days as a directional trend; the first "
                     "day may be partial (see hours_covered)."),
        }

    # ------------------------------------------------------------------
    # historical tools (2015-2020 + 2022-2025, 292 places)
    # ------------------------------------------------------------------
    def _resolve_city(self, name):
        key = str(name).strip().lower()
        key = ALIASES.get(key, key).lower()
        exact = [c for c in self.cities if c.lower() == key]
        if len(exact) == 1:
            return exact[0], None
        # same-named places in different states are stored as "Name (State)"
        prefixed = [c for c in self.cities if c.lower().startswith(key + " (")]
        found = exact or prefixed or search_city(key, self.cities)
        if len(found) == 1:
            return found[0], None
        if len(found) > 1:
            return None, {"error": f"'{name}' matches several places: {found[:12]}. Be more specific."}
        return None, {"error": (f"'{name}' is not one of the {len(self.cities)} places in the "
                                f"historical dataset (it covers {len(self.states)} states/UTs). "
                                f"Live AQI works for any place; history needs a covered one.")}

    def _city_summary(self, city):
        d = self.hist[self.hist["City"] == city]
        season = d[d["is_crop_burning_season"]]["AQI"]
        off = d[~d["is_crop_burning_season"]]["AQI"]
        by_month = d.groupby("month")["AQI"].mean()
        out = {
            "city": city,
            "state": str(d["State"].iloc[0]),
            "days_of_data": int(len(d)),
            "date_range": f"{d['Date'].min().date()} to {d['Date'].max().date()}",
            "avg_aqi": _r(d["AQI"].mean(), 0),
            "pct_days_poor_or_worse": _r((d["AQI"] > 200).mean() * 100, 0),
            "worst_month": MONTHS[int(by_month.idxmax()) - 1],
            "worst_month_avg_aqi": _r(by_month.max(), 0),
            "best_month": MONTHS[int(by_month.idxmin()) - 1],
            "best_month_avg_aqi": _r(by_month.min(), 0),
        }
        dom = d["dominant_pollutants"].dropna()
        if len(dom):
            out["most_common_dominant_pollutant"] = str(dom.value_counts().index[0])
        if len(season) and len(off):
            out["crop_burning_season_avg_aqi"] = _r(season.mean(), 0)
            out["rest_of_year_avg_aqi"] = _r(off.mean(), 0)
            out["crop_burning_increase_pct"] = _r((season.mean() / off.mean() - 1) * 100, 0)
        return out

    def get_city_history(self, city):
        name, err = self._resolve_city(city)
        if err:
            return err
        out = self._city_summary(name)
        out["note"] = HISTORY_NOTE
        return out

    def compare_cities(self, city_a, city_b):
        a, err = self._resolve_city(city_a)
        if err:
            return err
        b, err = self._resolve_city(city_b)
        if err:
            return err
        sa, sb = self._city_summary(a), self._city_summary(b)
        out = {
            a: sa, b: sb,
            "higher_avg_aqi_city": a if sa["avg_aqi"] >= sb["avg_aqi"] else b,
            "note": HISTORY_NOTE,
        }
        if "crop_burning_season_avg_aqi" in sa and "crop_burning_season_avg_aqi" in sb:
            out["higher_crop_season_aqi_city"] = (
                a if sa["crop_burning_season_avg_aqi"] >= sb["crop_burning_season_avg_aqi"] else b)
        return out

    def get_crop_burning_impact(self):
        from data_prep_india import CROP_BURNING_HOTSPOT_CITIES as HOT

        h = self.hist
        rows = []
        for c in HOT:
            if c in self.cities:
                s = self._city_summary(c)
                if "crop_burning_increase_pct" not in s:
                    continue
                rows.append({
                    "city": c,
                    "rest_of_year_avg_aqi": s["rest_of_year_avg_aqi"],
                    "crop_burning_season_avg_aqi": s["crop_burning_season_avg_aqi"],
                    "increase_pct": s["crop_burning_increase_pct"],
                })
        others = h[~h["City"].isin(HOT)]
        o_in = others[others["is_crop_burning_season"]]["AQI"].mean()
        o_off = others[~others["is_crop_burning_season"]]["AQI"].mean()
        seas = h[h["is_crop_burning_season"]]
        n = seas.groupby("City")["AQI"].count()
        top = (seas[seas["City"].isin(n[n >= MIN_SEASON_DAYS].index)].groupby("City")["AQI"]
               .mean().sort_values(ascending=False).head(5))
        return {
            "season_definition": ("15 Oct to 30 Nov each year, when paddy stubble burning in "
                                  "Punjab and Haryana sends smoke across North India."),
            "hotspot_cities": rows,
            "other_cities_avg": {
                "rest_of_year_avg_aqi": _r(o_off, 0),
                "crop_burning_season_avg_aqi": _r(o_in, 0),
                "increase_pct": _r((o_in / o_off - 1) * 100, 0),
            },
            "top5_cities_in_season": [{"city": c, "avg_aqi": _r(v, 0)} for c, v in top.items()],
            "note": HISTORY_NOTE,
        }

    def rank_places(self, metric="avg_aqi", state=None, order="worst", limit=10):
        """Rank places by a historical metric, optionally within one state."""
        if metric not in METRICS:
            return {"error": f"Unknown metric '{metric}'. Use one of {METRICS}."}
        if order not in ("worst", "best"):
            return {"error": "order must be 'worst' or 'best'."}
        limit = max(1, min(int(limit), 25))
        h = self.hist
        if state:
            match = [x for x in self.states if x.lower() == str(state).strip().lower()]
            if not match:
                return {"error": f"Unknown state '{state}'. Available: {self.states}"}
            state = match[0]
            h = h[h["State"] == state]
        days = h.groupby("City")["AQI"].count()
        if metric == "crop_season_aqi":
            s = h[h["is_crop_burning_season"]]
            n = s.groupby("City")["AQI"].count()
            ok = n[n >= MIN_SEASON_DAYS].index
            vals = s[s["City"].isin(ok)].groupby("City")["AQI"].mean()
        else:
            ok = days[days >= MIN_DAYS_RANKING].index
            g = h[h["City"].isin(ok)]
            vals = (g.groupby("City")["AQI"].mean() if metric == "avg_aqi"
                    else g.groupby("City")["AQI"].apply(lambda x: (x > 200).mean() * 100))
        if vals.empty:
            return {"error": "No place has enough data for that ranking."}
        vals = vals.sort_values(ascending=(order == "best")).head(limit)
        st_of = h.drop_duplicates("City").set_index("City")["State"]
        return {
            "metric": metric, "order": order, "state": state,
            "ranking": [{"rank": i + 1, "place": c, "state": str(st_of[c]),
                         "value": _r(v, 0), "days_of_data": int(days[c])}
                        for i, (c, v) in enumerate(vals.items())],
            "places_considered": int(len(ok)),
            "note": ("Only places with at least "
                     f"{MIN_SEASON_DAYS if metric == 'crop_season_aqi' else MIN_DAYS_RANKING} days of data "
                     "are ranked. Values mix both periods of the dataset (2015-2020 and 2022-2025) "
                     "for the 26 older cities. " + HISTORY_NOTE),
        }

    # ------------------------------------------------------------------
    # health guidance (static, CPCB-based -- so advice is grounded, not improvised)
    # ------------------------------------------------------------------
    def get_health_guidance(self, aqi_category, profile="general"):
        if aqi_category not in CATEGORY_IMPACT:
            return {"error": f"Unknown category '{aqi_category}'. Use one of {CATEGORIES}."}
        if profile not in PROFILE_EXTRAS:
            return {"error": f"Unknown profile '{profile}'. Use one of {PROFILES}."}
        out = {
            "aqi_category": aqi_category,
            "possible_health_impact": CATEGORY_IMPACT[aqi_category],
            "general_precautions": CATEGORY_PRECAUTIONS[aqi_category],
            "disclaimer": DISCLAIMER,
        }
        if PROFILE_EXTRAS[profile]:
            out["profile"] = profile
            out["profile_specific_advice"] = PROFILE_EXTRAS[profile]
        return out

    # ------------------------------------------------------------------
    # tool schemas sent to the LLM
    # ------------------------------------------------------------------
    def schemas(self):
        place = {"type": "string", "description": "City, town or place name, e.g. 'Delhi', 'Pune', 'London'."}
        city = {"type": "string",
                "description": "An Indian city/place in the historical dataset (292 places), e.g. 'Delhi', 'Lucknow', 'Pune'."}
        return [
            {
                "name": "get_current_aqi",
                "description": ("Get the CURRENT estimated air quality for any place in the world: "
                                "estimated AQI on the Indian CPCB scale, its category (Good to Severe) "
                                "and the live pollutant readings (PM2.5, PM10, NO2, SO2, O3, CO). "
                                "Use for 'how is the air in X right now / today'."),
                "input_schema": {"type": "object", "properties": {"place": place}, "required": ["place"]},
            },
            {
                "name": "get_aqi_forecast",
                "description": ("Get the estimated AQI FORECAST for a place for the next 1-4 days: per day "
                                "average and peak AQI, the worst hour and the cleanest hour, plus whether "
                                "air quality is improving or worsening. Use for 'tomorrow', 'this weekend', "
                                "'best time to go out' questions."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "place": place,
                        "days": {"type": "integer", "minimum": 1, "maximum": 4,
                                 "description": "Number of forecast days (default 3)."},
                    },
                    "required": ["place"],
                },
            },
            {
                "name": "get_city_history",
                "description": ("Historical daily-AQI summary (2015-2020 and 2022-2025) for one of 292 Indian "
                                "places: average AQI, worst/best months, share of Poor-or-worse days, "
                                "most common dominant pollutant and the crop-burning-season effect. "
                                "Use for 'typical' or 'usually' questions."),
                "input_schema": {"type": "object", "properties": {"city": city}, "required": ["city"]},
            },
            {
                "name": "compare_cities",
                "description": ("Compare the historical air quality of two of the 292 Indian places, "
                                "overall and during crop-burning season."),
                "input_schema": {
                    "type": "object",
                    "properties": {"city_a": city, "city_b": city},
                    "required": ["city_a", "city_b"],
                },
            },
            {
                "name": "get_crop_burning_impact",
                "description": ("How much crop (stubble) burning season (15 Oct - 30 Nov) raises AQI in "
                                "North Indian hotspot cities vs the rest of the year, plus the 5 most "
                                "affected cities. Takes no arguments."),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "rank_places",
                "description": ("Rank Indian places by historical air quality, optionally within one "
                                "state: metric 'avg_aqi', 'crop_season_aqi' or 'pct_poor_or_worse_days'; "
                                "order 'worst' (most polluted first) or 'best'. Use for 'most polluted "
                                "cities in Punjab', 'cleanest places in India' questions."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": METRICS},
                        "state": {"type": "string", "description": "Optional state/UT name, e.g. 'Punjab'."},
                        "order": {"type": "string", "enum": ["worst", "best"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 25},
                    },
                },
            },
            {
                "name": "get_health_guidance",
                "description": ("Get CPCB-based possible health impacts and general precautions for an AQI "
                                "category, optionally tailored to a person type. Call this after you know "
                                "the category whenever the user asks whether something is safe or what "
                                "precautions to take."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "aqi_category": {"type": "string", "enum": CATEGORIES},
                        "profile": {"type": "string", "enum": PROFILES,
                                    "description": "Who the advice is for (default 'general')."},
                    },
                    "required": ["aqi_category"],
                },
            },
        ]
