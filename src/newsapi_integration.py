#!/usr/bin/env python3
"""
NewsAPI integration for Ishtar AI
"""

import os
from typing import List, Dict, Any, Optional, Union
import logging
from dotenv import load_dotenv
import uuid
from tqdm import tqdm
from pydantic import BaseModel, Field, HttpUrl, validator
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import settings
from src.config import settings


# Define Pydantic models for NewsAPI data
class NewsSource(BaseModel):
    id: Optional[str] = None
    name: str


class NewsArticle(BaseModel):
    source: NewsSource
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: HttpUrl
    urlToImage: Optional[HttpUrl] = None
    publishedAt: datetime
    content: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class NewsAPIResponse(BaseModel):
    status: str
    totalResults: int
    articles: List[NewsArticle]

    @validator("status")
    def validate_status(cls, v):
        if v != "ok":
            raise ValueError(f"Expected status 'ok', got {v}")
        return v


class NewsAPIClient:
    """Client for interacting with NewsAPI"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize NewsAPI client with API key"""
        self.api_key = api_key or settings.newsapi_key
        self.client = None

        if not self.api_key:
            logger.warning("NewsAPI key not set. News fetching will be unavailable.")
            return

        try:
            # Import here to avoid requiring newsapi-python if not used
            from newsapi import NewsApiClient

            # Initialize NewsAPI client
            self.client = NewsApiClient(api_key=self.api_key)
            logger.info("NewsAPI client initialized")

        except ImportError:
            logger.warning(
                "newsapi-python package not installed. Run 'pip install newsapi-python' to enable news fetching."
            )
        except Exception as e:
            logger.error(f"Error initializing NewsAPI client: {e}")

    def get_articles(
        self,
        query: str = "war zone OR conflict OR frontline",
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get articles from NewsAPI

        Args:
            query: Search query
            language: Article language
            sort_by: Sorting method
            page_size: Number of articles to fetch

        Returns:
            List of articles
        """
        if not self.client:
            logger.warning("NewsAPI client not initialized")
            return []

        try:
            response = self.client.get_everything(
                q=query, language=language, sort_by=sort_by, page_size=page_size
            )

            # Validate the response with Pydantic
            try:
                validated_response = NewsAPIResponse(**response)
                # Convert Pydantic models to dicts for backward compatibility
                articles = [article.dict() for article in validated_response.articles]
                logger.info(
                    f"Fetched and validated {len(articles)} articles from NewsAPI"
                )
                return articles
            except Exception as validation_error:
                logger.warning(f"Response validation error: {validation_error}")
                # Fall back to unvalidated response if validation fails
                articles = response.get("articles", [])
                logger.info(
                    f"Fetched {len(articles)} articles from NewsAPI (unvalidated)"
                )
                return articles

        except Exception as e:
            logger.error(f"Error fetching articles from NewsAPI: {e}")
            return []


def chunk_articles(
    articles: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50
) -> tuple:
    """
    Split articles into chunks

    Args:
        articles: List of articles
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Tuple of (document chunks, metadata)
    """
    try:
        # Use updated import path
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            # Fallback to old import path
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        docs, metas = [], []
        for art in articles:
            # Use content, fallback to description, fallback to title
            text = (
                art.get("content") or art.get("description") or art.get("title") or ""
            )

            # Remove potential truncation artifacts
            if "[+" in text:
                text = text.split("[+")[0]

            # Skip empty texts
            if not text.strip():
                continue

            for chunk in splitter.split_text(text):
                docs.append(chunk)
                metas.append(
                    {
                        "source": art.get("source", {}).get("name"),
                        "publish_date": art.get("publishedAt"),  # ISO 8601
                        "author": art.get("author"),
                        "url": art.get("url"),
                        "title": art.get("title"),
                        "topic": "war-zones",
                        "content": chunk,  # Include the chunk text in metadata for retrieval
                    }
                )

        logger.info(f"Created {len(docs)} chunks from {len(articles)} articles")
        return docs, metas

    except ImportError:
        logger.error(
            "langchain package not installed. Run 'pip install langchain' to enable text splitting."
        )
        return [], []
    except Exception as e:
        logger.error(f"Error chunking articles: {e}")
        return [], []


def embed_chunks(
    docs: List[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
) -> List[List[float]]:
    """
    Embed document chunks using OpenAI

    Args:
        docs: List of document chunks
        api_key: OpenAI API key
        model: OpenAI embedding model to use
        dimensions: Embedding dimensions to use (1024 or 1536)

    Returns:
        List of embeddings
    """
    try:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not set. Set OPENAI_API_KEY in .env file.")
            return []

        import openai

        # Set API key
        openai.api_key = api_key

        # Select the appropriate model based on dimensions
        # Note: text-embedding-3-small actually produces 1536 dimensions
        if dimensions == 1024 or dimensions == 1536:
            if dimensions != 1536:
                logger.warning(
                    f"Requested {dimensions} dimensions, but using text-embedding-3-small which produces 1536 dimensions"
                )
            model = "text-embedding-3-small"  # 1536 dimensions
            actual_dimensions = 1536
        else:
            logger.warning(
                f"Unsupported dimension {dimensions}, using text-embedding-3-small (1536 dimensions)"
            )
            model = "text-embedding-3-small"
            actual_dimensions = 1536

        logger.info(
            f"Using embedding model {model} with {actual_dimensions} dimensions"
        )

        # Process in batches to avoid rate limits
        batch_size = 20
        all_embeddings = []

        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding chunks"):
            batch = docs[i : i + batch_size]
            try:
                response = openai.embeddings.create(input=batch, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                # Add empty embeddings to maintain index alignment
                all_embeddings.extend([[0.0] * actual_dimensions] * len(batch))

        logger.info(
            f"Created {len(all_embeddings)} embeddings with {actual_dimensions} dimensions"
        )
        return all_embeddings

    except ImportError:
        logger.error(
            "openai package not installed. Run 'pip install openai' to enable embeddings."
        )
        return []
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        return []


def index_to_pinecone(
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    pinecone_client=None,
    namespace: str = "news",
) -> bool:
    """
    Index embeddings to Pinecone

    Args:
        embeddings: List of embeddings
        metadata: List of metadata
        pinecone_client: Pinecone client
        namespace: Pinecone namespace

    Returns:
        Success status
    """
    # Import the client if not provided
    if not pinecone_client:
        from src.pinecone_integration import get_pinecone_client

        pinecone_client = get_pinecone_client()

    if not pinecone_client or not pinecone_client.index:
        logger.error("Pinecone client or index not initialized")
        return False

    # Prepare vectors for Pinecone
    vectors = []
    for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
        vectors.append({"id": str(uuid.uuid4()), "values": embedding, "metadata": meta})

    # Upload to Pinecone
    logger.info(f"Indexing {len(vectors)} vectors to Pinecone")
    try:
        batch_size = 100

        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
            batch = vectors[i : i + batch_size]
            success = pinecone_client.upsert(batch)
            if not success:
                logger.error(f"Failed to upload batch {i//batch_size + 1}")
                return False

        logger.info("Successfully indexed all vectors to Pinecone")
        return True

    except Exception as e:
        logger.error(f"Error indexing to Pinecone: {e}")
        return False


def fetch_and_index_news(
    query: str = "war zone OR conflict OR frontline",
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_dimensions: int = 1024,
) -> bool:
    """
    End-to-end pipeline to fetch news and index to Pinecone

    Args:
        query: Search query
        language: Article language
        sort_by: Sorting method
        page_size: Number of articles to fetch
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        embedding_dimensions: Dimensions for embeddings (1024 or 1536)

    Returns:
        Success status
    """
    # Initialize NewsAPI client
    news_client = NewsAPIClient()

    # Fetch articles
    logger.info(f"Fetching news articles for query: '{query}'")
    articles = news_client.get_articles(query, language, sort_by, page_size)

    if not articles:
        logger.error("No articles fetched")
        return False

    # Chunk articles
    logger.info("Chunking articles")
    docs, metas = chunk_articles(articles, chunk_size, chunk_overlap)

    if not docs:
        logger.error("No chunks created")
        return False

    # Embed chunks
    logger.info("Embedding chunks")
    embeddings = embed_chunks(docs, dimensions=embedding_dimensions)

    if not embeddings or len(embeddings) != len(docs):
        logger.error("Embedding failed or incomplete")
        return False

    # Index to Pinecone
    logger.info("Indexing to Pinecone")
    from src.pinecone_integration import get_pinecone_client

    pinecone_client = get_pinecone_client()

    return index_to_pinecone(embeddings, metas, pinecone_client)


if __name__ == "__main__":
    logger.info("Running NewsAPI integration test")

    # Test fetching articles
    news_client = NewsAPIClient()
    articles = news_client.get_articles(page_size=10)  # Fetch 10 for testing

    if articles:
        logger.info(f"Successfully fetched {len(articles)} articles")

        # Test chunking
        docs, metas = chunk_articles(articles)

        if docs:
            logger.info(f"Successfully created {len(docs)} chunks")

            # Test embedding - only if OpenAI API key is set
            if os.environ.get("OPENAI_API_KEY"):
                embeddings = embed_chunks(docs[:3])  # Only embed 3 for testing

                if embeddings:
                    logger.info(f"Successfully created {len(embeddings)} embeddings")

                    # Test indexing - only if Pinecone is configured
                    from src.pinecone_integration import get_pinecone_client

                    pinecone_client = get_pinecone_client()

                    if pinecone_client and pinecone_client.index:
                        success = index_to_pinecone(
                            embeddings, metas[:3], pinecone_client
                        )

                        if success:
                            logger.info("Successfully indexed test chunks to Pinecone")
                        else:
                            logger.error("Failed to index test chunks to Pinecone")
                    else:
                        logger.warning(
                            "Pinecone client not initialized, skipping indexing test"
                        )
                else:
                    logger.error("Failed to create embeddings")
            else:
                logger.warning("OpenAI API key not set, skipping embedding test")
        else:
            logger.error("Failed to create chunks")
    else:
        logger.error("Failed to fetch articles")
