from typing import Protocol

from app.schemas.chunks import RetrievedChunk


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class PassthroughReranker:
    """No-op reranker. Replace with a cross-encoder (Cohere Rerank, bge-reranker-v2)
    by implementing the same `rerank` signature."""

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return candidates
