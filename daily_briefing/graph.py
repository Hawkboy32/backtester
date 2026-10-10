"""The real LangGraph graph: activity and risk gather in parallel (neither
depends on the other), fan into the writer node - a genuine fan-out/fan-in
topology, not a disguised linear script. See activity_node.py/risk_node.py
for what each actually reads, writer_node.py for how the briefing gets
written.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from activity_node import gather_activity
from risk_node import gather_risk
from writer_node import write_briefing


class BriefingState(TypedDict, total=False):
    activity: dict
    risk: dict
    briefing: str
    provider: str
    model: str | None


def _activity_step(state: BriefingState) -> dict:
    return {"activity": gather_activity()}


def _risk_step(state: BriefingState) -> dict:
    return {"risk": gather_risk()}


def _writer_step(state: BriefingState) -> dict:
    kwargs: dict = {"provider": state.get("provider", "ollama")}
    model = state.get("model")
    if model:
        kwargs["claude_model" if kwargs["provider"] == "claude" else "model"] = model
    briefing = write_briefing(state["activity"], state["risk"], **kwargs)
    return {"briefing": briefing}


def build_graph():
    graph = StateGraph(BriefingState)
    graph.add_node("activity", _activity_step)
    graph.add_node("risk", _risk_step)
    graph.add_node("writer", _writer_step)
    graph.add_edge(START, "activity")
    graph.add_edge(START, "risk")
    graph.add_edge(["activity", "risk"], "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def run_briefing(*, provider: str = "ollama", model: str | None = None) -> BriefingState:
    compiled = build_graph()
    return compiled.invoke({"provider": provider, "model": model})
