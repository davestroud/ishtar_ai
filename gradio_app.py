"""Gradio front-end for Ishtar AI.

Styling & layout focused on a newsroom-friendly chat experience.
"""

import asyncio
import gradio as gr

from ishtar_ai.rag.pipeline import query_pipeline


async def _chat_fn(
    message: str, history: list[tuple[str, str]]
):  # gr.ChatInterface signature
    """Handle a single user message and return assistant reply."""

    answer = await query_pipeline(message)
    return answer


DESCRIPTION = """
### The Ishtar AI Initiative  
*Supercharging conflict journalism with real-time intelligence.*

Ishtar AI blends vetted knowledge from Pinecone with live web context
from Tavily and the reasoning power of Meta's Llama 4.  Tailored for
Fox journalists reporting from war zones and humanitarian crises, it
delivers grounded answers, sources & context in seconds.
"""


demo = gr.ChatInterface(
    fn=_chat_fn,
    title="Ishtar AI",
    description=DESCRIPTION,
    theme=gr.themes.Soft(),
    examples=[
        "What is the latest UN statement on the Gaza humanitarian corridor?",
        "How many people have been displaced in Sudan since January 2024?",
        "Summarise today's most credible sources on the cholera outbreak in Yemen.",
    ],
)


if __name__ == "__main__":
    demo.launch()
