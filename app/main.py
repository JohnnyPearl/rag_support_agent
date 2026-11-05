from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, qa
from app.config import get_settings
from app.dependencies import AppState
from app.generation.graph import build_graph
from app.generation.llm import OpenAIChat
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.reranker import PassthroughReranker
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store.qdrant import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key, model=settings.openai_embedding_model
    )
    store = QdrantVectorStore(path=settings.qdrant_path, collection=settings.collection_name)
    if not store.client.collection_exists(settings.collection_name):
        raise RuntimeError(
            f"Qdrant collection '{settings.collection_name}' not found at "
            f"{settings.qdrant_path}. Run `python -m scripts.ingest <pdf>` first."
        )
    retriever = DenseRetriever(embedder, store)
    chat = OpenAIChat(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    graph = build_graph(retriever, PassthroughReranker())

    app.state.app_state = AppState(
        embedder=embedder,
        store=store,
        retriever=retriever,
        chat=chat,
        graph=graph,
        top_k=settings.top_k,
    )
    try:
        yield
    finally:
        store.client.close()


app = FastAPI(title="RAG User Manual Q&A", lifespan=lifespan)
app.include_router(health.router)
app.include_router(qa.router)
