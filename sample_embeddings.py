#!/usr/bin/env python3
import requests
import json
import numpy as np
from ollama_client import OllamaClient
import time


class OllamaEmbeddings:
    def __init__(self, base_url=None, model="llama3"):
        self.client = OllamaClient(base_url)
        self.model = model
        self.api_url = self.client.api_url

    def get_embedding(self, text):
        """Get embeddings for a single text string"""
        payload = {
            "model": self.model,
            "prompt": text,
        }

        response = requests.post(f"{self.api_url}/embeddings", json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", [])
        else:
            raise Exception(f"Failed to get embedding: {response.text}")

    def batch_embeddings(self, texts, batch_size=10):
        """Process a batch of texts and return their embeddings"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                # Rate limiting to be nice to the API
                time.sleep(0.1)

            embeddings.extend(batch_embeddings)
            print(f"Processed {i+len(batch)}/{len(texts)} texts")

        return embeddings

    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar(self, query_embedding, embeddings_list, texts=None):
        """Find the most similar embeddings to a query embedding"""
        if not query_embedding or not embeddings_list:
            return []

        similarities = []
        for i, emb in enumerate(embeddings_list):
            sim = self.cosine_similarity(query_embedding, emb)
            similarities.append((i, sim))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # If texts are provided, return (text, similarity) pairs
        if texts and len(texts) == len(embeddings_list):
            return [(texts[i], sim) for i, sim in similarities]

        return similarities


if __name__ == "__main__":
    # Demo of using embeddings
    embedder = OllamaEmbeddings(model="llama3")

    # Sample texts
    texts = [
        "The cat sat on the mat.",
        "The dog played in the yard.",
        "I love machine learning and artificial intelligence.",
        "Neural networks are powerful tools for deep learning.",
        "The weather is nice today, perfect for a walk.",
        "Python is a popular programming language for data science.",
    ]

    print("Getting embeddings for sample texts...")
    embeddings = embedder.batch_embeddings(texts)

    # Demo query
    query = "AI and deep learning technologies"
    print(f"\nFinding texts most similar to: '{query}'")

    query_embedding = embedder.get_embedding(query)
    results = embedder.find_most_similar(query_embedding, embeddings, texts)

    print("\nResults (ranked by similarity):")
    for text, similarity in results:
        print(f"{similarity:.4f} - {text}")

    # Example: Computing similarity between two texts directly
    text1 = "Quantum computing is an emerging technology."
    text2 = "Quantum computers use qubits instead of classical bits."

    print("\nComputing similarity between two texts:")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")

    emb1 = embedder.get_embedding(text1)
    emb2 = embedder.get_embedding(text2)

    similarity = embedder.cosine_similarity(emb1, emb2)
    print(f"Similarity: {similarity:.4f}")
