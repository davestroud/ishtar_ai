"""Llama Gradio App for Hugging Face Spaces"""

import os
import gradio as gr
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Hugging Face Inference API configuration
MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
API_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def generate_response(message, history, temperature=0.7, max_tokens=256):
    """Generate a response from Llama."""
    if not message.strip():
        return "Please enter a message to generate a response."

    if not API_TOKEN:
        return "Please set your Hugging Face API token as a secret in Hugging Face Spaces settings."

    # Format messages for the API
    messages = []

    # Convert Gradio's chatbot format to the API's format
    if history:
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append(f"User: {user_msg}")
            if bot_msg:
                messages.append(f"Assistant: {bot_msg}")

    # Add the current message
    messages.append(f"User: {message}\nAssistant:")

    try:
        prompt = "\n".join(messages)

        payload = {
            "inputs": prompt,
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens},
        }

        response = httpx.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Typical response: list[{generated_text: str}]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            generated = data[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt) :]
            return generated.strip()
        return str(data)

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Ishstar AI App") as demo:
    gr.Markdown("# Ishtar AI Chat")
    gr.Markdown(f"Have a conversation with {MODEL_ID}")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=400,
                show_copy_button=True,
                label="Conversation",
                type="messages",  # Use OpenAI-style message format
            )

            msg = gr.Textbox(
                placeholder="Type your message here...",
                label="Your message",
                show_label=False,
                container=False,
            )

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"
            )

            max_tokens = gr.Slider(
                minimum=50, maximum=1000, value=256, step=50, label="Max Tokens"
            )

    def respond(message, chat_history, temp, tokens):
        bot_message = generate_response(message, chat_history, temp, tokens)
        chat_history.append((message, bot_message))
        return "", chat_history

    # Use the same respond function for both button click and Enter key
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[msg, chatbot],
    )

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[msg, chatbot],
    )

    clear_btn.click(lambda: ("", []), None, [msg, chatbot])

    # Default examples
    gr.Examples(
        examples=[
            "Tell me a short story about a robot learning to paint.",
            "Explain quantum computing in simple terms.",
            "Write a haiku about the changing seasons.",
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()
