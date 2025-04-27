#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone constants
PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY",
    "pcsk_519mbc_UcUkNV6mASAXkW9zDLnVjcA7UwVmmvaEFUvsKFtxNRsSRkD2vEiEgsW2Vo24JDV",
)
PINECONE_HOST = os.environ.get(
    "PINECONE_HOST", "ishtar-tpk95ap.svc.aped-4627-b74a.pinecone.io"
)
INDEX_NAME = "ishtar"


class PineconeClient:
    """Client for interacting with Pinecone vector database"""

    def __init__(self, api_key: Optional[str] = None, host: Optional[str] = None):
        """Initialize Pinecone client with API key and host"""
        self.api_key = api_key or PINECONE_API_KEY
        self.host = host or PINECONE_HOST
        self.index_name = INDEX_NAME
        self.index = None

        if not self.api_key:
            logger.warning(
                "Pinecone API key not set. Vector search capabilities will be unavailable."
            )
            return None

        try:
            # Import here to avoid requiring pinecone-client if not used
            from pinecone import Pinecone, ServerlessSpec

            # Initialize Pinecone
            pc = Pinecone(api_key=self.api_key)

            # List indexes to verify connection
            indexes = pc.list_indexes()

            # Connect to existing index
            self.index = pc.Index(host=self.host)
            logger.info(
                f"Connected to Pinecone index: {self.index_name} at {self.host}"
            )

        except ImportError:
            logger.warning(
                "pinecone-client package not installed. Run 'pip install pinecone-client' to enable vector search."
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {e}")

    def query(
        self, vector: List[float], top_k: int = 5, include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone index with a vector

        Args:
            vector: Embedding vector to query with
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results

        Returns:
            List of matching documents with scores and metadata
        """
        if not self.index:
            logger.warning("Pinecone index not initialized")
            return []

        try:
            results = self.index.query(
                vector=vector, top_k=top_k, include_metadata=include_metadata
            )

            # Format results
            matches = []
            for match in results.get("matches", []):
                matches.append(
                    {
                        "id": match.get("id", ""),
                        "score": match.get("score", 0),
                        "metadata": match.get("metadata", {}),
                    }
                )

            return matches

        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []

    def upsert(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Insert or update vectors in the Pinecone index

        Args:
            vectors: List of vectors with format [{"id": "id1", "values": [0.1, 0.2, ...], "metadata": {...}}, ...]

        Returns:
            Success status
        """
        if not self.index:
            logger.warning("Pinecone index not initialized")
            return False

        try:
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            return False


def get_pinecone_client() -> Optional[PineconeClient]:
    """Get a Pinecone client instance"""
    return PineconeClient()


if __name__ == "__main__":
    # Test Pinecone integration
    client = get_pinecone_client()

    if client and client.index:
        print(f"Successfully connected to Pinecone index: {client.index_name}")

        # Test query with a random vector (for testing purposes only)
        import random

        test_vector = [
            random.random() for _ in range(1536)
        ]  # OpenAI embedding dimension

        results = client.query(test_vector, top_k=3)
        if results:
            print(f"Query returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  ID: {result['id']}")
                print(f"  Score: {result['score']}")
                print(f"  Metadata: {result['metadata']}")
        else:
            print("No results found in index or query failed")
    else:
        print("Failed to initialize Pinecone client")
