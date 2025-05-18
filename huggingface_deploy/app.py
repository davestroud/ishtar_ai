"""Llama Gradio App for Hugging Face Spaces"""

import os
import gradio as gr
from dotenv import load_dotenv
from llama_api_client import LlamaAPIClient

# Load environment variables
load_dotenv()

# Llama API configuration
API_KEY = os.getenv("LLAMA_API_KEY")
if not API_KEY:
    # For Hugging Face Spaces, you should set this as a secret
    # If using HF_TOKEN, adjust accordingly
    API_KEY = os.getenv("HF_TOKEN")

MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Initialize Llama client
client = LlamaAPIClient(
    api_key=API_KEY,
    base_url="https://api.llama.com/v1/",
)


def generate_response(message, history, temperature=0.7, max_tokens=256):
    """Generate a response from Llama."""
    if not message.strip():
        return "Please enter a message to generate a response."

    if not API_KEY:
        return (
            "Please set your Llama API key as a secret in Hugging Face Spaces settings."
        )

    # Format messages for the API
    messages = []

    # Convert Gradio's chatbot format to the API's format
    if history:
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    # Add the current message
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=temperature,
        )

        # Extract the completion message content
        if hasattr(response, "completion_message"):
            if hasattr(response.completion_message, "content"):
                if hasattr(response.completion_message.content, "text"):
                    return response.completion_message.content.text
                elif isinstance(response.completion_message.content, str):
                    return response.completion_message.content

        # Return full response as last resort
        return str(response)

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
