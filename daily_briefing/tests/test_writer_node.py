from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import writer_node

_ACTIVITY = {"running": True, "trades_today": 2, "last_signal": None, "last_error": None, "per_ticker_signals": [], "closed_trades_today": []}
_RISK = {"accounts": [], "sizing_mode": "pct_equity", "sizing_value": 20.0}


def test_write_with_ollama_sends_both_structured_blocks(monkeypatch):
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"message": {"content": "Chopper made 2 trades today."}}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(writer_node.urllib.request, "urlopen", fake_urlopen)

    briefing = writer_node.write_briefing(_ACTIVITY, _RISK)

    assert briefing == "Chopper made 2 trades today."
    user_message = captured["payload"]["messages"][1]["content"]
    assert "trades_today" in user_message
    assert "sizing_mode" in user_message


def test_write_with_claude_uses_the_real_sdk_shape(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Chopper made 2 trades today.")])

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    )

    briefing = writer_node.write_briefing(_ACTIVITY, _RISK, provider="claude")

    assert briefing == "Chopper made 2 trades today."
    assert captured["model"] == writer_node.DEFAULT_CLAUDE_MODEL
    assert captured["system"] == writer_node._SYSTEM_PROMPT
