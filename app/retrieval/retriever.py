from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.vector_store.base import VectorStore
from app.schemas.chunks import RetrievedChunk


class DenseRetriever:
    """Dense top-k retrieval. Hybrid (sparse + dense) and parent-doc expansion
    are intentional future hooks — add them as new retriever classes implementing
    the same call signature."""

    def __init__(self, embedder: OpenAIEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vector = await self.embedder.aembed_query(query)
        return self.store.search(vector, top_k)
