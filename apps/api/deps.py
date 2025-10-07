from ishtar.config.settings import settings
from ishtar.rag.vectorstore import make_vectorstore
from ishtar.rag.retriever import Retriever
from ishtar.agents.graph import build_graph

_vectorstore = make_vectorstore(settings.vector_backend)
_retriever = Retriever(_vectorstore, k=settings.retrieve_k, rerank_top_k=settings.rerank_top_k)
_graph = build_graph()

def get_vectorstore():
    return _vectorstore

def get_retriever():
    return _retriever

def get_graph():
    return _graph
