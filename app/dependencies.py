from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.generation.llm import OpenAIChat
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store.qdrant import QdrantVectorStore


@dataclass
class AppState:
    embedder: OpenAIEmbedder
    store: QdrantVectorStore
    retriever: DenseRetriever
    chat: OpenAIChat
    graph: Any
    top_k: int


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
