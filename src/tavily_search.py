#!/usr/bin/env python3
import requests
import json
import os
import sys
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

        # Base URL for Tavily API
        self.base_url = "https://api.tavily.com"

        # Print debug info
        print(
            f"Tavily API client initialized with key: {self.api_key[:5]}...",
            file=sys.stderr,
        )
        print(f"Using API endpoint: {self.base_url}", file=sys.stderr)

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
        max_tokens: int = 8000,
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

        print(f"Making API request to: {url}", file=sys.stderr)
        print(f"Query: {query}", file=sys.stderr)

        # Build the payload
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
            # Added verbose debug information
            print("Sending request to Tavily API...", file=sys.stderr)

            # Skip the status check since it can cause errors
            # Make the actual request with proper authorization headers
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=15)
                print(
                    f"Received response with status code: {response.status_code}",
                    file=sys.stderr,
                )
            except requests.exceptions.Timeout:
                print("Tavily API request timed out", file=sys.stderr)
                return {"error": "The web search request timed out. Please try again."}
            except requests.exceptions.ConnectionError:
                print("Tavily API connection error", file=sys.stderr)
                return {
                    "error": "Could not connect to the web search service. Please check your internet connection."
                }
            except Exception as e:
                print(f"Tavily API request error: {e}", file=sys.stderr)
                return {"error": f"Web search request error: {str(e)}"}

            # Process the response
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "results" in result:
                        print(
                            f"Success! Found {len(result['results'])} search results",
                            file=sys.stderr,
                        )
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}", file=sys.stderr)
                    print(f"Response content: {response.text[:500]}", file=sys.stderr)
                    return {"error": f"Could not parse search results: {str(e)}"}
            elif response.status_code == 401 or response.status_code == 403:
                return {
                    "error": "API key is invalid or expired. Please check your Tavily API key."
                }
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please try again later."}
            else:
                try:
                    error_details = response.json()
                    print(f"API Error: {error_details}", file=sys.stderr)
                    return {
                        "error": f"API error ({response.status_code}): {error_details}"
                    }
                except:
                    print(f"API Error: {response.text}", file=sys.stderr)
                    return {
                        "error": f"API error ({response.status_code}): {response.text}"
                    }

        except Exception as e:
            print(f"Exception in Tavily search: {e}", file=sys.stderr)
            return {"error": f"Web search failed: {str(e)}"}

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
