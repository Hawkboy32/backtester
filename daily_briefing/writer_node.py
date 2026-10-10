"""Writer node: turns the activity + risk nodes' structured output into a
plain-English daily briefing. Same dual-provider shape used across the
other three CV projects built this same session - Ollama by default
(free, verified live), Claude via provider="claude" (real Anthropic SDK,
code-complete and mocked-tested only - no ANTHROPIC_API_KEY was available
when this was built).
"""

from __future__ import annotations

import json
import urllib.request

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"

_SYSTEM_PROMPT = (
    "You write a short daily briefing for the owner of an automated trading system "
    "called Chopper, from two JSON blocks: today's trading activity and the current "
    "real risk/exposure across every linked account. Write 3-6 plain-English "
    "sentences - what happened today, current exposure, and anything that needs "
    "attention (a breaker tripped, an account error, high exposure). Never invent "
    "numbers not present in the data. If an account shows an error, mention it - do "
    "not silently drop it. You are reporting, not deciding anything or placing any "
    "trade yourself."
)


def _user_message(activity: dict, risk: dict) -> str:
    return (
        f"Today's activity:\n{json.dumps(activity, indent=2)}\n\n"
        f"Current risk/exposure:\n{json.dumps(risk, indent=2)}\n\n"
        "Write the briefing now."
    )


def write_briefing(
    activity: dict,
    risk: dict,
    *,
    provider: str = "ollama",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    anthropic_api_key: str | None = None,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
) -> str:
    if provider == "claude":
        return _write_with_claude(activity, risk, api_key=anthropic_api_key, model=claude_model)
    return _write_with_ollama(activity, risk, ollama_url=ollama_url, model=model)


def _write_with_ollama(activity: dict, risk: dict, *, ollama_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(activity, risk)},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:  # noqa: S310 - fixed local Ollama URL, not user input
        data = json.loads(resp.read().decode("utf-8"))
    return (data["message"]["content"] or "").strip()


def _write_with_claude(activity: dict, risk: dict, *, api_key: str | None, model: str) -> str:
    import anthropic  # noqa: PLC0415 - optional dependency, only needed for provider="claude"

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(activity, risk)}],
        output_config={"effort": "low"},  # a short daily summary - not worth paying for more
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()
