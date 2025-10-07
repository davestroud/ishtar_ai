from ishtar.rag.vectorstore import make_vectorstore
from ishtar.config.settings import settings
from ishtar.ingestion.readers.rss import pull_rss
from ishtar.ingestion.pipeline import ingest_items

if __name__ == "__main__":
    vs = make_vectorstore(settings.vector_backend)
    items = pull_rss("https://news.un.org/feed/subscribe/en/news/all/rss.xml")
    ingest_items(items, vs)
    print(f"Ingested {len(items)} items into index '{settings.index_name}'.")
