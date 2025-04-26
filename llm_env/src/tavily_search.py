#!/usr/bin/env python3
import requests
import os
import json
from typing import List, Dict, Any, Optional


class TavilySearch:
    """Client for Tavily Search API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from parameter or environment variable"""
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable or pass to constructor."
            )

        self.base_url = "https://api.tavily.com"

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
        include_answer: bool = True,
        include_images: bool = False,
        include_raw_content: bool = False,
        max_tokens: int = 12000,
    ) -> Dict[str, Any]:
        """
        Perform a search using the Tavily API

        Args:
            query: The search query
            search_depth: 'basic' (faster) or 'advanced' (more thorough but slower)
            max_results: Number of results to return (1-10)
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            include_answer: Whether to include an AI-generated answer
            include_images: Whether to include image results
            include_raw_content: Whether to include raw content of search results
            max_tokens: Maximum number of tokens in the response

        Returns:
            Dict containing search results
        """

        url = f"{self.base_url}/search"

        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_images": include_images,
            "include_raw_content": include_raw_content,
            "max_tokens": max_tokens,
        }

        # Add optional parameters if provided
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        # Set up headers with proper authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Tavily API request failed: {str(e)}")

    def answer_question(self, question: str, max_results: int = 3) -> str:
        """
        Quick helper to just get an answer to a question with context

        Args:
            question: The question to answer
            max_results: Number of results to use for context

        Returns:
            String containing the answer
        """
        try:
            result = self.search(
                query=question,
                max_results=max_results,
                include_answer=True,
                search_depth="advanced",
            )

            answer = result.get("answer", "")
            if not answer:
                context = "\n\n".join(
                    [
                        f"Source {i+1}: {item.get('content', '')}"
                        for i, item in enumerate(result.get("results", []))
                    ]
                )
                return f"No direct answer found, but here's some relevant information:\n\n{context}"

            return answer
        except Exception as e:
            return f"Error retrieving information: {str(e)}"


if __name__ == "__main__":
    # Simple test if run directly
    import sys

    # Check if API key exists
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY environment variable not set")
        sys.exit(1)

    # Create client
    tavily = TavilySearch(api_key)

    # Test search
    test_query = "What is the latest news about artificial intelligence?"
    print(f"Searching for: {test_query}")
    try:
        result = tavily.answer_question(test_query)
        print("\nAnswer:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
