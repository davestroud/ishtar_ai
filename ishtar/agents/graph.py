from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .prompts import REFINE_TMPL, SUMMARIZE_TMPL, VERIFY_TMPL
from ishtar.llm.client import llm_call


class AgentState(TypedDict, total=False):
    query: str
    context: list[dict[str, Any]]
    draft: str
    verdict: str
    final: str


def _context_to_prompt(ctx: list[dict[str, Any]] | None) -> str:
    if not ctx:
        return "[]"
    return json.dumps(ctx, indent=2)


def summarize(state: AgentState) -> AgentState:
    ctx = _context_to_prompt(state.get("context"))
    q = state["query"]
    return {"draft": llm_call(SUMMARIZE_TMPL.format(query=q, context=ctx))}


def verify(state: AgentState) -> AgentState:
    draft = state["draft"]
    ctx = _context_to_prompt(state.get("context"))
    out = llm_call(VERIFY_TMPL.format(draft=draft, context=ctx))
    return {"verdict": out}


def refine(state: AgentState) -> AgentState:
    draft, verdict = state["draft"], state["verdict"]
    return {"final": llm_call(REFINE_TMPL.format(draft=draft, verdict=verdict))}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("summarize", summarize)
    g.add_node("verify", verify)
    g.add_node("refine", refine)
    g.add_edge("summarize", "verify")
    g.add_edge("verify", "refine")
    g.set_entry_point("summarize")
    g.add_edge("refine", END)
    return g.compile()
