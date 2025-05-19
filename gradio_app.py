"""Stylish Gradio front-end for Ishtar AI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import gradio as gr

from ishtar_ai.rag.pipeline import query_pipeline

# ---------------------------------------------------------------------------
# Simple light theme & minimal CSS ------------------------------------------
# ---------------------------------------------------------------------------

theme = gr.themes.Default()

custom_css = """
/* Header strip */
#header {background:#003366;padding:10px;display:flex;align-items:center;}
#header img{height:40px;margin-right:10px;}
#header h1{font-size:1.5em;margin:0;color:#ffffff;}

/* Chat bubbles */
#chatbot .user{background:#eef1f6;color:#000000;}
#chatbot .bot{background:#f4f6fa;color:#000000;}
#chatbot .user,#chatbot .bot{border-radius:8px;padding:10px;margin:5px 0;}
"""

# ---------------------------------------------------------------------------
# Back-end glue --------------------------------------------------------------
# ---------------------------------------------------------------------------


async def _answer(
    user_msg: str, history: list[tuple[str, str]], use_live: bool, k_docs: int
) -> str:
    """Delegate to RAG pipeline with adjustable top-k."""

    # We pass `top_k` based on slider; `use_live` could toggle Tavily in future.
    # For now, simply call query_pipeline.
    answer: str = await query_pipeline(user_msg, top_k=int(k_docs))
    return answer


# ---------------------------------------------------------------------------
# Build UI -------------------------------------------------------------------
# ---------------------------------------------------------------------------

with gr.Blocks(theme=theme, css=custom_css, title="Ishtar AI") as demo:
    # Header branding
    with gr.Row(elem_id="header"):
        logo_path: str | None = None
        for candidate in (Path("ishtar_logo.png"), Path("logo.png")):
            if candidate.exists():
                logo_path = str(candidate)
                break
        if logo_path:
            gr.Image(value=logo_path, show_label=False)
        gr.HTML("<h1>Ishtar AI – Newsroom Chat</h1>")

    with gr.Column(scale=1):
        gr.Markdown("## Settings")
        live_toggle = gr.Checkbox(label="Real-time context (Tavily)", value=True)
        num_docs_slider = gr.Slider(
            1, 5, step=1, value=3, label="Number of context documents"
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot", height=550, show_copy_button=True, show_share_button=False
    )

    chat_interface = gr.ChatInterface(
        fn=_answer,
        chatbot=chatbot,
        additional_inputs=[live_toggle, num_docs_slider],
        title="",
        description=(
            "<p style='margin-top:0'>Supercharging conflict journalism with real-time intelligence. "
            "Blends vetted Pinecone knowledge with live web context and Meta Llama 4 reasoning.</p>"
        ),
        # (streaming handled server-side; older Gradio versions use fn return)
    )


if __name__ == "__main__":  # pragma: no cover
    demo.queue().launch()
