#!/usr/bin/env python3
"""
Shared text processing utilities for the Ishtar AI system
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap

    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Split by paragraphs
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Keep the overlap from the end of the previous chunk
            overlap_text = (
                " ".join(current_chunk.split()[-overlap:]) if overlap > 0 else ""
            )
            current_chunk = overlap_text + " " + para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n" + para
            else:
                current_chunk = para

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_langchain_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Get a LangChain text splitter

    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        LangChain text splitter or None if not available
    """
    try:
        # Use updated import path
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            # Fallback to old import path
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except ImportError:
        logger.error(
            "langchain package not installed. Run 'pip install langchain' to enable text splitting."
        )
        return None
    except Exception as e:
        logger.error(f"Error creating text splitter: {e}")
        return None
