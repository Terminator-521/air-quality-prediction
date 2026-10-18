"""
Tests for the AQI chat assistant.

No network and no API key needed:
  - live Open-Meteo calls are replaced by a fake that returns realistic data
  - the Anthropic API is replaced by a scripted fake client
The trained models and the historical CPCB data are the real ones.

Run from the project folder:   python -m pytest tests -q
"""
import json
import os
import types
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import pytest

import anthropic
try:                      # newer SDKs use httpx2, older ones httpx
    import httpx2 as httpx
except ImportError:
    import httpx
import assistant
import live_data
from assistant_tools import AirQualityTools, CATEGORIES
from data_prep_india import load_history

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- fixtures
def fake_fetch_factory(counter):
    def fake_fetch(lat, lon):
        counter["fetch"] = counter.get("fetch", 0) + 1
        now = pd.Timestamp.now().floor("h")
        times = pd.date_range(now - pd.Timedelta(hours=47), periods=24 * 6, freq="h")
        n = len(times)
        wave = np.sin((times.hour.values - 6) / 24 * 2 * np.pi)   # peaks in the evening
        return pd.DataFrame({
            "time": times,
            "PM2.5": 90 + 60 * wave, "PM10": 160 + 80 * wave,
            "CO": 1.0 + 0.4 * wave, "NO2": 40 + 10 * wave,
            "SO2": 12.0, "O3": 45.0,
            "is_forecast": times > now,
        })
    return fake_fetch


def fake_geocode(query, count=5):
    if query.lower() == "nowhere":
        return []
    return [
        {"name": query.title(), "admin1": "Test State", "country": "India",
         "latitude": 28.7, "longitude": 77.1},
        {"name": query.title(), "admin1": "Other", "country": "Elsewhere",
         "latitude": 1.0, "longitude": 2.0},
    ][:count]


@pytest.fixture(scope="module")
def parts():
    return (
        joblib.load(ROOT / "models/india_aqi_classifier.joblib"),
        joblib.load(ROOT / "models/india_aqi_regressor.joblib"),
        joblib.load(ROOT / "models/india_features.joblib"),
        load_history(str(ROOT / "data/city_day.csv"), str(ROOT / "data/aqi_daily_2022_2025.csv.gz")),
    )


@pytest.fixture()
def tools(parts):
    clf, reg, feats, hist = parts
    counter = {}
    t = AirQualityTools(clf, reg, feats, hist,
                        geocode_fn=fake_geocode, fetch_fn=fake_fetch_factory(counter))
    t.counter = counter
    return t


# ----------------------------------------------------------------- tools: live
def test_current_aqi(tools):
    out = tools.call("get_current_aqi", {"place": "delhi"})
    assert "error" not in out
    assert out["category"] in CATEGORIES
    assert 0 < out["estimated_aqi"] < 1000
    assert set(out["pollutants_ug_m3"]) == {"PM2.5", "PM10", "NO2", "SO2", "O3"}
    assert out["other_possible_matches"]          # ambiguity is surfaced
    json.dumps(out)                               # must be JSON-serialisable


def test_forecast(tools):
    out = tools.call("get_aqi_forecast", {"place": "delhi", "days": 2})
    assert "error" not in out
    assert len(out["daily_forecast"]) == 2
    d = out["daily_forecast"][0]
    assert d["peak_aqi"] >= d["avg_aqi"] >= 0
    assert len(d["worst_hour"]) == 5 and d["worst_hour"].endswith(":00")
    assert out["trend"]["direction"] in {"worsening", "improving", "roughly steady"}
    json.dumps(out)


def test_forecast_days_are_clamped(tools):
    assert len(tools.call("get_aqi_forecast", {"place": "x", "days": 99})["daily_forecast"]) <= 4
    assert len(tools.call("get_aqi_forecast", {"place": "x", "days": 0})["daily_forecast"]) == 1


def test_live_lookups_are_cached(tools):
    tools.call("get_current_aqi", {"place": "pune"})
    tools.call("get_aqi_forecast", {"place": "pune"})
    assert tools.counter["fetch"] == 1


# ----------------------------------------------------------------- tools: history
def test_city_history_real_data(tools):
    out = tools.call("get_city_history", {"city": "Delhi"})
    assert out["city"] == "Delhi"
    assert out["crop_burning_increase_pct"] > 30        # the project's headline finding
    assert out["worst_month"] in {"Oct", "Nov", "Dec", "Jan"}


def test_city_partial_and_unknown(tools):
    assert tools.call("get_city_history", {"city": "luck"})["city"] == "Lucknow"
    err = tools.call("get_city_history", {"city": "Atlantis"})
    assert "error" in err and "292" in err["error"]     # says what the dataset covers


def test_aliases_and_same_named_places(tools):
    assert tools.call("get_city_history", {"city": "Gurgaon"})["city"] == "Gurugram"
    assert tools.call("get_city_history", {"city": "Bangalore"})["city"] == "Bengaluru"
    err = tools.call("get_city_history", {"city": "Aurangabad"})   # Bihar vs Maharashtra
    assert "error" in err and "Aurangabad (Bihar)" in err["error"]
    ok = tools.call("get_city_history", {"city": "Aurangabad (Maharashtra)"})
    assert ok["state"] == "Maharashtra"


def test_rank_places(tools):
    out = tools.call("rank_places", {"metric": "avg_aqi", "state": "Punjab", "limit": 3})
    assert "error" not in out and len(out["ranking"]) <= 3
    assert all(r["state"] == "Punjab" for r in out["ranking"])
    vals = [r["value"] for r in out["ranking"]]
    assert vals == sorted(vals, reverse=True)           # worst first
    best = tools.call("rank_places", {"order": "best", "limit": 5})["ranking"]
    assert [r["value"] for r in best] == sorted(r["value"] for r in best)
    assert all(r["days_of_data"] >= 365 for r in best)  # sparse places excluded
    assert "error" in tools.call("rank_places", {"metric": "nope"})
    assert "error" in tools.call("rank_places", {"state": "Atlantis"})
    season = tools.call("rank_places", {"metric": "crop_season_aqi", "limit": 5})
    assert "error" not in season and season["ranking"][0]["value"] > 150


def test_compare_cities(tools):
    out = tools.call("compare_cities", {"city_a": "Delhi", "city_b": "Mumbai"})
    assert "error" not in out
    assert out["higher_avg_aqi_city"] == "Delhi"
    assert out["higher_crop_season_aqi_city"] == "Delhi"


def test_crop_burning_impact(tools):
    out = tools.call("get_crop_burning_impact", {})
    assert len(out["hotspot_cities"]) >= 5
    assert len(out["top5_cities_in_season"]) == 5
    assert all(r["increase_pct"] > 0 for r in out["hotspot_cities"])


# ----------------------------------------------------------------- tools: health + safety
def test_health_guidance(tools):
    out = tools.call("get_health_guidance", {"aqi_category": "Very Poor", "profile": "asthma_or_lung_disease"})
    assert "inhaler" in out["profile_specific_advice"]
    assert "not medical advice" in out["disclaimer"]
    assert "error" in tools.call("get_health_guidance", {"aqi_category": "Spicy"})
    assert "error" in tools.call("get_health_guidance", {"aqi_category": "Good", "profile": "alien"})


def test_call_never_raises(tools, parts):
    assert "error" in tools.call("no_such_tool", {})
    assert "error" in tools.call("get_current_aqi", {})                    # missing arg
    assert "error" in tools.call("get_current_aqi", {"place": "nowhere"})  # geocode: no match

    def boom(*a, **k):
        raise live_data.LiveDataError("Couldn't reach the air quality service (test).")
    clf, reg, feats, hist = parts
    t = AirQualityTools(clf, reg, feats, hist, geocode_fn=fake_geocode, fetch_fn=boom)
    assert "Couldn't reach" in t.call("get_current_aqi", {"place": "delhi"})["error"]

    def weird(*a, **k):
        raise RuntimeError("secret internal detail")
    t = AirQualityTools(clf, reg, feats, hist, geocode_fn=fake_geocode, fetch_fn=weird)
    err = t.call("get_current_aqi", {"place": "delhi"})["error"]
    assert "secret internal detail" not in err      # internals are not leaked to the model/user


def test_schemas_match_dispatch(tools):
    names = {s["name"] for s in tools.schemas()}
    assert names == set(tools._dispatch)
    for s in tools.schemas():
        assert s["input_schema"]["type"] == "object" and s["description"]


# ----------------------------------------------------------------- agent loop (scripted fake API)
def text_block(t):
    return types.SimpleNamespace(type="text", text=t)

def tool_block(i, name, inp):
    return types.SimpleNamespace(type="tool_use", id=i, name=name, input=inp)

def resp(content, stop):
    return types.SimpleNamespace(content=content, stop_reason=stop)

class FakeClient:
    """Stands in for anthropic.Anthropic. Replays a scripted list of responses; once the
    script runs out it keeps asking for a tool (to exercise the step limit)."""
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self            # so client.messages.create(...) works

    def create(self, **kw):
        # snapshot like the real SDK does: the request is serialised at call time,
        # before the agent loop appends anything else to its messages list
        self.calls.append(dict(kw, messages=list(kw["messages"])))
        if self.script:
            return self.script.pop(0)
        return resp([tool_block("loop", "get_crop_burning_impact", {})], "tool_use")


def test_agent_tool_then_answer(tools):
    client = FakeClient([
        resp([text_block("Let me check."), tool_block("t1", "get_current_aqi", {"place": "delhi"})], "tool_use"),
        resp([text_block("Delhi is Poor.")], "end_turn"),
    ])
    r = assistant.run_assistant(client, "claude-haiku-5-5", [], "How is Delhi?", tools)
    assert r.text == "Delhi is Poor." and r.steps == 2
    assert [t["tool"] for t in r.trace] == ["get_current_aqi"]
    # the tool result went back to the model, linked to the right tool_use id
    last = client.calls[1]["messages"][-1]
    assert last["role"] == "user"
    assert last["content"][0]["tool_use_id"] == "t1"
    assert last["content"][0]["is_error"] is False
    assert "estimated_aqi" in last["content"][0]["content"]
    # tools + system prompt (with today's date) were sent
    assert client.calls[0]["tools"] and "Today's date is" in client.calls[0]["system"]


def test_agent_parallel_calls_one_user_message(tools):
    client = FakeClient([
        resp([tool_block("a", "get_city_history", {"city": "Delhi"}),
              tool_block("b", "get_city_history", {"city": "Mumbai"})], "tool_use"),
        resp([text_block("Done.")], "end_turn"),
    ])
    r = assistant.run_assistant(client, "m", [], "q", tools)
    results = client.calls[1]["messages"][-1]["content"]
    assert [x["tool_use_id"] for x in results] == ["a", "b"]
    assert len(r.trace) == 2


def test_agent_marks_tool_errors(tools):
    client = FakeClient([
        resp([tool_block("e", "get_city_history", {"city": "Atlantis"})], "tool_use"),
        resp([text_block("I can't find that city.")], "end_turn"),
    ])
    assistant.run_assistant(client, "m", [], "q", tools)
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_agent_step_limit(tools):
    client = FakeClient([])        # model asks for a tool forever
    r = assistant.run_assistant(client, "m", [], "q", tools, max_steps=3)
    assert len(client.calls) == 3 and r.steps == 3
    assert "one thing at a time" in r.text


def test_agent_keeps_history_and_handles_empty_text(tools):
    client = FakeClient([resp([], "end_turn")])
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    r = assistant.run_assistant(client, "m", hist, "again", tools)
    sent = client.calls[0]["messages"]
    assert sent[0]["content"] == "hi" and sent[-1]["content"] == "again"
    assert "rephrase" in r.text


def test_friendly_errors():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    auth = anthropic.AuthenticationError("bad", response=httpx.Response(401, request=req), body=None)
    assert "key was rejected" in assistant.friendly_error(auth)
    conn = anthropic.APIConnectionError(request=req)
    assert "internet" in assistant.friendly_error(conn)
    rate = anthropic.RateLimitError("slow", response=httpx.Response(429, request=req), body=None)
    assert "Rate limit" in assistant.friendly_error(rate)
    assert "ValueError" in assistant.friendly_error(ValueError("x"))


def test_api_key_resolution(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert assistant.get_api_key("  sk-typed ") == "sk-typed"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert assistant.get_api_key("") == "sk-env"
    assert assistant.get_api_key("sk-typed") == "sk-typed"     # UI box wins
