#!/usr/bin/env python3
"""
Shared embedding utilities for the Ishtar AI system
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_openai_embeddings(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 20,
    show_progress: bool = True,
) -> List[List[float]]:
    """
    Get embeddings from OpenAI's API

    Args:
        texts: List of texts to embed
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: OpenAI embedding model to use
        batch_size: Number of texts to embed in each API call
        show_progress: Whether to show a progress bar

    Returns:
        List of embeddings
    """
    try:
        import openai

        # Check if API key is set
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not set. Set OPENAI_API_KEY in .env file.")
            return []

        openai.api_key = api_key

        # Process in batches to avoid rate limits
        all_embeddings = []

        # Create iterator with or without progress bar
        batch_iterator = range(0, len(texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc="Creating embeddings")

        for i in batch_iterator:
            batch = texts[i : i + batch_size]
            try:
                response = openai.embeddings.create(input=batch, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                # Add empty embeddings to maintain index alignment
                # Use actual dimensionality of model - 1536 for text-embedding-3-small
                dims = 1536 if model == "text-embedding-3-small" else 1024
                all_embeddings.extend([[0.0] * dims] * len(batch))

        logger.info(f"Created {len(all_embeddings)} embeddings")
        return all_embeddings

    except ImportError:
        logger.error(
            "openai package not installed. Run 'pip install openai' to enable embeddings."
        )
        return []
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return []
