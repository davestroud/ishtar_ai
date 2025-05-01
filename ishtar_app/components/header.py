#!/usr/bin/env python3
import streamlit as st


def render_header():
    """Render the application header section"""
    st.title("Ishtar AI")
    st.markdown(
        """
    Welcome to Ishtar AI - your intelligent assistant powered by local models.
    """
    )

    with st.expander("About Ishtar AI"):
        st.markdown(
            """
        Ishtar AI is a local LLM interface with various integrations:
        - **Hugging Face**: Local model inference
        - **Tavily**: Web search capabilities
        - **LangSmith**: Tracing for analytics
        - **Pinecone**: Vector database for knowledge retrieval
        - **Weather API**: Current weather information
        
        Use the sidebar to configure model settings and enable/disable features.
        """
        )
