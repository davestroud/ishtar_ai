import gradio as gr
import asyncio
from ishtar_ai.rag.pipeline import query_pipeline

async def answer_query(query: str) -> str:
    return await query_pipeline(query)

iface = gr.Interface(fn=lambda q: asyncio.run(answer_query(q)), inputs="text", outputs="text", title="Ishtar AI")

if __name__ == "__main__":
    iface.launch()
