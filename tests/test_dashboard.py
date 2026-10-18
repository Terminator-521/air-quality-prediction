"""
End-to-end tests of the AQI Assistant tab, driven through Streamlit's AppTest.
Network and the Anthropic API are faked; the dashboard, tools and models are real.

Run from the project folder:   python -m pytest tests -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import anthropic
import assistant
import live_data
from test_assistant import (FakeClient, fake_fetch_factory, fake_geocode,
                            resp, text_block, tool_block)

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.chdir(ROOT)                                   # dashboard uses relative paths
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(live_data, "geocode", fake_geocode)
    monkeypatch.setattr(live_data, "fetch_air_quality", fake_fetch_factory({}))
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def test_loads_with_four_tabs(app):
    assert [t.label for t in app.tabs][-1] == "🤖 AQI Assistant"
    assert len(app.tabs) == 4
    assert any("API key" in i.value for i in app.info)        # prompts for a key


def test_without_key_asks_for_one(app):
    ask(app, "How is Delhi?")
    assert not app.exception
    assert any("API key" in w.value for w in app.warning)
    assert app.session_state["chat_history"] == []            # nothing sent to the LLM


def test_full_chat_with_tool_call(app, monkeypatch):
    fake = FakeClient([
        resp([tool_block("t1", "get_current_aqi", {"place": "Delhi"})], "tool_use"),
        resp([text_block("Delhi's air is currently estimated as shown above.")], "end_turn"),
    ])
    monkeypatch.setattr(assistant, "make_client", lambda key: fake)

    app.sidebar.text_input[0].set_value("sk-test-key").run()
    ask(app, "How is Delhi?")

    assert not app.exception, app.exception
    roles = [m.name for m in app.chat_message]
    assert roles == ["user", "assistant"]
    assert "currently estimated" in app.chat_message[1].markdown[0].value
    assert any("1 tool call used" in e.label for e in app.expander)
    assert app.session_state["chat_history"][-1]["role"] == "assistant"
    # the model was given the tools and the selected model string
    assert fake.calls[0]["model"] == "claude-haiku-5-5" and fake.calls[0]["tools"]

    # a follow-up keeps the conversation context
    fake.script = [resp([text_block("Mumbai is cleaner.")], "end_turn")]
    ask(app, "And Mumbai?")
    assert not app.exception
    sent = fake.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert len(app.chat_message) == 4


def test_sample_question_button(app, monkeypatch):
    fake = FakeClient([resp([text_block("Here you go.")], "end_turn")])
    monkeypatch.setattr(assistant, "make_client", lambda key: fake)
    app.sidebar.text_input[0].set_value("sk-test-key").run()
    app.button(key="sample_0").click().run()
    assert not app.exception
    assert fake.calls[0]["messages"][-1]["content"] == assistant.SAMPLE_QUESTIONS[0]


def test_bad_key_shows_friendly_error(app, monkeypatch):
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    class Rejecting:
        messages = None
        def __init__(self):
            self.messages = self
        def create(self, **kw):
            raise anthropic.AuthenticationError("bad", response=httpx.Response(401, request=req), body=None)

    monkeypatch.setattr(assistant, "make_client", lambda key: Rejecting())
    app.sidebar.text_input[0].set_value("sk-wrong").run()
    ask(app, "How is Delhi?")
    assert not app.exception
    assert any("key was rejected" in e.value for e in app.error)
    assert app.session_state["chat_history"] == []            # failed turn isn't replayed to the LLM


def clear_button(app):
    return [b for b in app.button if "Clear chat" in b.label]


def test_clear_button_appears_right_after_first_answer(app, monkeypatch):
    """Regression: the button used to be drawn before the first message was saved,
    so it only showed up after a second interaction."""
    fake = FakeClient([resp([text_block("Hi.")], "end_turn")])
    monkeypatch.setattr(assistant, "make_client", lambda key: fake)
    assert not clear_button(app)                      # nothing to clear yet
    app.sidebar.text_input[0].set_value("sk-test-key").run()
    ask(app, "hello")
    assert not app.exception, app.exception
    assert len(clear_button(app)) == 1


def test_clear_chat(app, monkeypatch):
    fake = FakeClient([resp([text_block("Hi.")], "end_turn")])
    monkeypatch.setattr(assistant, "make_client", lambda key: fake)
    app.sidebar.text_input[0].set_value("sk-test-key").run()
    ask(app, "hello")
    assert len(app.chat_message) == 2
    clear_button(app)[0].click().run()
    assert not app.exception
    assert app.session_state["chat_display"] == [] and app.session_state["chat_history"] == []
    assert len(app.chat_message) == 0 and not clear_button(app)
    assert len(fake.calls) == 1                       # clearing doesn't call the API


# ----------------------------------------------------------------- history / places
def _select(at, key, value):
    next(s for s in at.selectbox if s.key == key).set_value(value).run()
    assert not at.exception, at.exception


def test_history_state_filter_and_place_without_coordinates(app):
    _select(app, "hist_state", "Punjab")
    places = next(s for s in app.selectbox if s.key == "hist_city").options
    assert 5 < len(places) < 60 and "Amritsar" in places
    _select(app, "hist_state", "Chhattisgarh")
    _select(app, "hist_city", "Tumidih")        # a place with no map coordinate
    assert any("No map coordinate" in c.value for c in app.caption)


def test_history_same_named_places_and_live_pick(app):
    _select(app, "hist_city", "Aurangabad (Bihar)")
    _select(app, "live_pick", "Pune")           # pick an Indian place instead of typing
    assert not any("Couldn't" in e.value for e in app.error)
    assert any("Historical context for Pune" in c.value for c in app.caption)
