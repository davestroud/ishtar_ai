from __future__ import annotations
from langgraph.graph import StateGraph, END
from .prompts import SUMMARIZE_TMPL, VERIFY_TMPL, REFINE_TMPL
from ishtar.llm.client import llm_call

def summarize(state):
    ctx = state["context"]; q = state["query"]
    return {"draft": llm_call(SUMMARIZE_TMPL.format(query=q, context=ctx))}

def verify(state):
    draft = state["draft"]; ctx = state["context"]
    out = llm_call(VERIFY_TMPL.format(draft=draft, context=ctx))
    return {"verdict": out}

def refine(state):
    draft, verdict = state["draft"], state["verdict"]
    return {"final": llm_call(REFINE_TMPL.format(draft=draft, verdict=verdict))}

def build_graph():
    g = StateGraph()
    g.add_node("summarize", summarize)
    g.add_node("verify", verify)
    g.add_node("refine", refine)
    g.add_edge("summarize", "verify")
    g.add_edge("verify", "refine")
    g.set_entry_point("summarize")
    g.add_edge("refine", END)
    return g.compile()
