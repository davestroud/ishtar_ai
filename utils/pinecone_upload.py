#!/usr/bin/env python3
"""
Utility script for uploading documents to Pinecone
Usage: python utils/pinecone_upload.py path/to/documents
"""

import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Pinecone integration
from src.pinecone_integration import get_pinecone_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Get embeddings from OpenAI's API"""
    try:
        import openai

        # Check if API key is set
        if not openai.api_key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not found in environment variables")

        # Create embeddings in batches (max 1000 per API call)
        batch_size = 100  # Smaller batch size to avoid timeouts
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

            response = openai.embeddings.create(input=batch_texts, model=model)

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    except ImportError:
        print("Error: openai package not installed. Install with 'pip install openai'")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        sys.exit(1)


def read_file(file_path: Path) -> tuple:
    """Read a file and extract its content and extension"""
    try:
        extension = file_path.suffix.lower()

        # Read text files
        if extension in [
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".csv",
            ".json",
            ".xml",
            ".py",
            ".js",
            ".java",
            ".c",
            ".cpp",
        ]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content, extension

        # Read PDF files
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                return content, extension
            except ImportError:
                print(
                    "Error: pypdf package not installed. Install with 'pip install pypdf'"
                )
                return None, extension

        # Read DOCX files
        elif extension == ".docx":
            try:
                import docx

                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
                return content, extension
            except ImportError:
                print(
                    "Error: python-docx package not installed. Install with 'pip install python-docx'"
                )
                return None, extension

        else:
            print(f"Unsupported file type: {extension}")
            return None, extension

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
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


def upload_to_pinecone(
    client, vectors: List[Dict[str, Any]], batch_size: int = 100
) -> bool:
    """Upload vectors to Pinecone in batches"""
    if not client or not client.index:
        print("Pinecone client or index not initialized")
        return False

    success = True

    # Process in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        print(
            f"Uploading batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size} ({len(batch)} vectors)"
        )

        try:
            result = client.upsert(batch)
            if not result:
                print(f"Error uploading batch {i//batch_size + 1}")
                success = False
        except Exception as e:
            print(f"Error uploading batch: {e}")
            success = False

    return success


def process_files(
    directory_path: str, chunk_size: int = 1000, overlap: int = 200
) -> None:
    """Process all files in a directory and upload to Pinecone"""
    path = Path(directory_path)

    # Check if path exists
    if not path.exists():
        print(f"Error: Path '{path}' does not exist")
        sys.exit(1)

    # Get Pinecone client
    client = get_pinecone_client()

    if not client or not client.index:
        print("Error: Could not initialize Pinecone client or index")
        sys.exit(1)

    # List of files to process
    files = []

    # Handle different path types
    if path.is_file():
        files = [path]
    elif path.is_dir():
        # Get all text, PDF, and DOCX files recursively
        files = list(path.glob("**/*.txt"))
        files.extend(path.glob("**/*.md"))
        files.extend(path.glob("**/*.pdf"))
        files.extend(path.glob("**/*.docx"))
        files.extend(path.glob("**/*.html"))
        files.extend(path.glob("**/*.htm"))
        files.extend(path.glob("**/*.json"))

    if not files:
        print(f"No supported files found in '{path}'")
        sys.exit(1)

    print(f"Found {len(files)} files to process")

    # Process files
    all_chunks = []
    chunk_metadata = []

    for file_path in files:
        print(f"Processing file: {file_path}")
        content, extension = read_file(file_path)

        if not content:
            print(f"Skipping file (no content): {file_path}")
            continue

        # Generate chunks
        chunks = chunk_text(content, chunk_size, overlap)

        # Create metadata for each chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append(
                {
                    "title": file_path.name,
                    "source": str(file_path),
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "extension": extension,
                    "content": (
                        chunk[:300] + "..." if len(chunk) > 300 else chunk
                    ),  # Preview of content
                }
            )

    if not all_chunks:
        print("No content chunks were created. Exiting.")
        sys.exit(1)

    print(f"Created {len(all_chunks)} content chunks")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = get_openai_embeddings(all_chunks)

    if len(embeddings) != len(all_chunks):
        print(
            f"Error: Number of embeddings ({len(embeddings)}) does not match number of chunks ({len(all_chunks)})"
        )
        sys.exit(1)

    # Prepare vectors for upload
    vectors = []
    for i, (chunk, embedding, metadata) in enumerate(
        zip(all_chunks, embeddings, chunk_metadata)
    ):
        vectors.append(
            {"id": str(uuid.uuid4()), "values": embedding, "metadata": metadata}
        )

    # Upload to Pinecone
    print(
        f"Uploading {len(vectors)} vectors to Pinecone index '{client.index_name}'..."
    )
    success = upload_to_pinecone(client, vectors)

    if success:
        print("Upload complete!")
    else:
        print("Upload completed with errors. Check the logs above.")


def main():
    parser = argparse.ArgumentParser(description="Upload documents to Pinecone")
    parser.add_argument("path", help="Path to file or directory to process")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)",
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Overlap between chunks (default: 200)"
    )

    args = parser.parse_args()

    process_files(args.path, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
