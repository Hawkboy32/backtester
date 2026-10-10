from __future__ import annotations

import graph


def test_graph_fans_out_then_fans_in_to_the_writer(monkeypatch):
    """Mocks all three real nodes - this test is purely about the graph's
    wiring (does activity+risk really run before writer, does the writer
    see both outputs), not about what any node actually reads."""
    monkeypatch.setattr(graph, "gather_activity", lambda: {"trades_today": 7})
    monkeypatch.setattr(graph, "gather_risk", lambda: {"sizing_mode": "pct_equity"})

    captured = {}

    def fake_write_briefing(activity, risk, **kwargs):
        captured["activity"] = activity
        captured["risk"] = risk
        captured["kwargs"] = kwargs
        return "a briefing"

    monkeypatch.setattr(graph, "write_briefing", fake_write_briefing)

    result = graph.run_briefing(provider="ollama")

    assert result["briefing"] == "a briefing"
    assert captured["activity"] == {"trades_today": 7}
    assert captured["risk"] == {"sizing_mode": "pct_equity"}
    assert captured["kwargs"] == {"provider": "ollama"}


def test_graph_passes_a_model_override_to_the_writer(monkeypatch):
    monkeypatch.setattr(graph, "gather_activity", lambda: {})
    monkeypatch.setattr(graph, "gather_risk", lambda: {})
    captured = {}
    monkeypatch.setattr(graph, "write_briefing", lambda activity, risk, **kw: captured.update(kw) or "ok")

    graph.run_briefing(provider="claude", model="claude-opus-5-5")

    assert captured == {"provider": "claude", "claude_model": "claude-opus-5-5"}
