#!/usr/bin/env python3
"""
Command-line utility to fetch news articles and index them in Pinecone
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
import hashlib
import uuid
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from retrieval.newsapi_integration import NewsAPIClient
from utils.text_processing import get_langchain_splitter
from utils.embeddings import get_openai_embeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, validator, root_validator
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("news_fetcher.log")],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Define Pydantic models for validation
class ArticleMetadata(BaseModel):
    title: str
    description: Optional[str] = None
    url: HttpUrl
    published_at: Optional[str] = None
    source: str
    query: str
    content: str


class NewsArticle(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    url: HttpUrl
    publishedAt: Optional[str] = None
    content: Optional[str] = None
    source: Dict[str, Any]

    class Config:
        extra = "allow"


class FetcherSettings(BaseModel):
    query: str = Field(default="war zone OR conflict OR frontline")
    page_size: int = Field(default=10, ge=1, le=100)
    save_only: bool = Field(default=False)
    dimensions: Optional[int] = Field(default=None)
    verbose: bool = Field(default=False)

    @validator("dimensions")
    def validate_dimensions(cls, v):
        if v is not None and v not in [1024, 1536]:
            logger.warning(f"Dimensions {v} not supported, will use auto-detection")
        return v

    class Config:
        validate_assignment = True


def save_articles_to_file(articles: List[Dict[str, Any]], output_file: str) -> bool:
    """Save articles to a JSON file"""
    try:
        # Validate articles with Pydantic
        validated_articles = []
        for article in articles:
            try:
                validated_article = NewsArticle(**article)
                validated_articles.append(validated_article.dict())
            except Exception as e:
                logger.warning(f"Article validation error: {e}")
                validated_articles.append(article)  # Keep original if validation fails

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(validated_articles, f, indent=2)
        logger.info(f"Saved {len(articles)} articles to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving articles to file: {e}")
        return False


def save_news_articles(
    query: str, page_size: int, output_file: Optional[str] = None
) -> bool:
    """Fetch and save news articles without indexing them"""
    if output_file is None:
        output_file = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    logger.info(f"Fetching articles for query: '{query}'")
    news_client = NewsAPIClient()
    articles = news_client.get_articles(
        query=query,
        page_size=page_size,
    )

    if not articles:
        logger.error("No articles fetched")
        return False

    # Save articles to file
    success = save_articles_to_file(articles, output_file)

    if success:
        logger.info(f"Successfully saved {len(articles)} articles to {output_file}")
        return True
    else:
        logger.error("Failed to save articles")
        return False


def get_pinecone_index_dims() -> int:
    """Get the dimensions of the Pinecone index"""
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pinecone_index = os.environ.get("PINECONE_INDEX")
        pinecone_host = os.environ.get("PINECONE_HOST")

        if not pinecone_api_key or not pinecone_index or not pinecone_host:
            logger.warning(
                "Missing Pinecone credentials, using default dimensions (1536)"
            )
            return 1536

        # New Pinecone client initialization
        from pinecone import Pinecone

        pc = Pinecone(api_key=pinecone_api_key)

        # Connect to the index using host
        index = pc.Index(host=pinecone_host)
        index_stats = index.describe_index_stats()

        # Extract dimensions from index stats
        if hasattr(index_stats, "dimension"):
            return index_stats.dimension
        elif "dimension" in index_stats:
            return index_stats["dimension"]
        else:
            logger.warning(
                "Could not detect dimensions from Pinecone index, using default (1536)"
            )
            return 1536

    except Exception as e:
        logger.warning(
            f"Error getting Pinecone index dimensions: {e}. Using default (1536)"
        )
        return 1536


def fetch_and_index_news(query: str, page_size: int, dims: int = 1024) -> bool:
    """Fetch news articles and index them in Pinecone"""
    logger.info(f"Fetching and indexing articles for query: '{query}'")

    # Initialize embedding model
    model_name = "text-embedding-3-small"  # This model supports 1024 dimensions
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("Missing OpenAI API key")
            return False

        embedding_model = OpenAIEmbeddings(model=model_name, dimensions=dims)
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        return False

    # Initialize Pinecone
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pinecone_index_name = os.environ.get("PINECONE_INDEX")
        pinecone_host = os.environ.get("PINECONE_HOST")

        if not pinecone_api_key or not pinecone_index_name or not pinecone_host:
            logger.error("Missing Pinecone credentials")
            return False

        # New Pinecone client initialization
        from pinecone import Pinecone

        pc = Pinecone(api_key=pinecone_api_key)

        # Connect to the index using host
        index = pc.Index(host=pinecone_host)
        logger.info(
            f"Connected to Pinecone index: {pinecone_index_name} at {pinecone_host}"
        )
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return False

    # Fetch news articles
    news_client = NewsAPIClient()
    articles = news_client.get_articles(
        query=query,
        page_size=page_size,
    )

    if not articles:
        logger.error("No articles fetched")
        return False

    logger.info(f"Fetched {len(articles)} articles")

    # Index articles in Pinecone
    try:
        count = 0
        batch_size = 10  # Process in small batches

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            vectors = []

            for article in batch:
                # Create content from article
                content = f"Title: {article.get('title', '')}\n"
                content += f"Description: {article.get('description', '')}\n"
                content += f"Content: {article.get('content', '')}\n"

                # Generate embedding
                embedding = embedding_model.embed_query(content)

                # Create metadata
                metadata = {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "query": query,
                }

                # Create unique ID
                article_id = f"news_{hashlib.md5(article.get('url', str(uuid.uuid4())).encode()).hexdigest()}"

                vectors.append(
                    {"id": article_id, "values": embedding, "metadata": metadata}
                )

            # Upsert batch to Pinecone
            index.upsert(vectors=vectors)
            count += len(batch)
            logger.info(f"Indexed {count}/{len(articles)} articles")

        logger.info(f"Successfully indexed {count} articles in Pinecone")
        return True

    except Exception as e:
        logger.error(f"Error indexing articles in Pinecone: {e}")
        return False


def main() -> None:
    """Main function to fetch and save news articles."""
    parser = argparse.ArgumentParser(description="Fetch and index news articles")
    parser.add_argument(
        "--query",
        type=str,
        default="war zone OR conflict OR frontline",
        help="Query to search for news articles",
    )
    parser.add_argument(
        "--page-size", type=int, default=10, help="Number of articles to fetch per page"
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Only save articles to JSON, skip embedding and indexing",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Dimensions for the embeddings (auto-detected from Pinecone index if not specified)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate arguments with Pydantic
    try:
        settings = FetcherSettings(
            query=args.query,
            page_size=args.page_size,
            save_only=args.save_only,
            dimensions=args.dimensions,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)

    if settings.verbose:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

    # Auto-detect dimensions if not specified
    if settings.dimensions is None:
        settings.dimensions = get_pinecone_index_dims()

    if settings.save_only:
        save_news_articles(settings.query, settings.page_size)
    else:
        success = fetch_and_index_news(
            settings.query, settings.page_size, settings.dimensions
        )
        if not success:
            logger.error("News fetching and indexing failed.")
            sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
