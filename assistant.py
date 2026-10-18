"""
assistant.py
------------
The AQI chat assistant: an LLM (Claude, via the Anthropic API) that answers
air-quality questions by CALLING TOOLS (see assistant_tools.py) instead of
guessing.

How it works (a standard tool-calling / "agent" loop, no framework needed):

    1. Send the conversation + the list of tools to the model.
    2. If the model replies with stop_reason == "tool_use", run each tool it
       asked for and send the results back as tool_result blocks.
    3. Repeat until the model answers in plain text (or we hit max_steps).

Why Claude with native tool use rather than an agent framework: the tool set
is small and fixed, so the loop is ~20 lines, easy to explain and debug, and
there is one dependency (the `anthropic` SDK).
"""

import datetime
import json
import os
from dataclasses import dataclass, field

import anthropic

# Label shown in the UI -> API model string
MODELS = {
    "Claude Haiku 5.5 (fast, low cost)": "claude-haiku-5-5",
    "Claude Sonnet 5.5 (smarter)": "claude-sonnet-5-5",
}
DEFAULT_MODEL_LABEL = "Claude Haiku 5.5 (fast, low cost)"

SYSTEM_PROMPT = """You are the AQI Assistant inside an air-quality dashboard. Today's date is {today}.

How to work:
- Never state an AQI, pollutant level or forecast from memory. Get it from a tool. If a tool returns an error, say what failed and suggest an alternative (for example, historical data for the 292 Indian places in the dataset works without a live lookup).
- The AQI values are ESTIMATES on India's CPCB scale, produced by a Random Forest model (trained on 2015-2020 CPCB station data; the 2022-2025 data is daily AQI only and is used for history and rankings, not for the model) applied to Open-Meteo pollutant data. Say "estimated" and do not claim they are official readings.
- For "is it safe to ..." or "what precautions ..." questions: look up the current AQI (and the forecast if the user means a later time), then call get_health_guidance with the resulting category and the person's profile (asthma, child, runner, ...). Start with a direct answer (go / take care / avoid), then give 2-4 short reasons or precautions using the numbers you retrieved.
- For timing questions ("best time to go out") use the forecast tool's cleanest and worst hours.
- Keep answers short and plain. Use the actual numbers. Light formatting only.
- You are not a doctor. For medical conditions or symptoms, recommend following their doctor's advice.
- Stay on air quality, pollution and related health or seasonal questions. Politely decline anything unrelated.
"""

SAMPLE_QUESTIONS = [
    "How is the air in Delhi right now?",
    "Is it safe for my asthmatic father to walk in Lucknow tomorrow morning?",
    "Compare Delhi and Mumbai during crop burning season",
    "Which are the 5 most polluted places in Punjab?",
    "When is the best time to go for a run in Pune today?",
]


@dataclass
class AssistantResult:
    text: str
    trace: list = field(default_factory=list)  # [{"tool", "input", "output"}, ...]
    steps: int = 0


def get_api_key(user_input=""):
    """Key from the UI box first, then the ANTHROPIC_API_KEY env var, then Streamlit secrets."""
    if user_input and user_input.strip():
        return user_input.strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY", "") or ""
    except Exception:
        return ""


def make_client(api_key):
    return anthropic.Anthropic(api_key=api_key)


def run_assistant(client, model, history, user_text, tools, max_steps=6, max_tokens=1024):
    """
    Answer one user message.

    history: earlier turns as plain [{"role": "user"|"assistant", "content": str}, ...].
             Tool calls from earlier turns are not replayed (the assistant's text
             answer already carries the key numbers), which keeps requests small.
    """
    system = SYSTEM_PROMPT.format(today=datetime.date.today().strftime("%A, %d %B %Y"))
    messages = list(history) + [{"role": "user", "content": user_text}]
    trace = []

    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools.schemas(),
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not text:
                text = "I couldn't put together an answer to that. Could you rephrase the question?"
            return AssistantResult(text=text, trace=trace, steps=step)

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            out = tools.call(block.name, dict(block.input))
            trace.append({"tool": block.name, "input": dict(block.input), "output": out})
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(out),
                "is_error": "error" in out,
            })
        messages.append({"role": "user", "content": results})

    return AssistantResult(
        text=("That question needed more lookups than I'm allowed in one go. "
              "Try asking about one place or one thing at a time."),
        trace=trace,
        steps=max_steps,
    )


def friendly_error(e):
    """Turn an Anthropic SDK exception into a message a user can act on."""
    if isinstance(e, anthropic.AuthenticationError):
        return "The API key was rejected. Check it and try again."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key doesn't have permission to use that model."
    if isinstance(e, anthropic.NotFoundError):
        return "That model isn't available for this API key. Pick a different model in the sidebar."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limit reached. Wait a moment and try again."
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your internet connection."
    if isinstance(e, anthropic.BadRequestError):
        detail = getattr(e, "message", "") or str(e)
        return f"The API rejected the request: {detail[:200]}"
    if isinstance(e, anthropic.APIStatusError):
        return f"The Anthropic API returned an error (status {e.status_code}). Try again shortly."
    return f"Something went wrong ({type(e).__name__}). Try again."
