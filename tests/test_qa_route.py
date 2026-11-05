from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import health, qa
from app.dependencies import AppState, get_state
from app.generation.graph import build_graph
from app.schemas.chunks import Chunk, RetrievedChunk


class FakeRetriever:
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        chunks = [
            Chunk(
                chunk_id=f"00000000-0000-5000-8000-00000000000{i}",
                text=f"context block {i} relevant to: {query}",
                page=10 + i,
                section_path=["Manual", f"Sec{i}"],
            )
            for i in range(3)
        ]
        return [RetrievedChunk(chunk=c, score=1.0 - 0.1 * i) for i, c in enumerate(chunks)]


class PassthroughReranker:
    async def rerank(self, query, candidates):
        return candidates


class FakeChat:
    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        for token in ["The", " answer", " is", " documented", " on", " page", " 10", "."]:
            yield token


@dataclass
class FakeStore:
    collection: str = "test"

    def count(self) -> int:
        return 3


@pytest.fixture
def app() -> FastAPI:
    retriever = FakeRetriever()
    chat = FakeChat()
    graph = build_graph(retriever, PassthroughReranker())
    state = AppState(
        embedder=None,  # type: ignore[arg-type]
        store=FakeStore(),  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        chat=chat,  # type: ignore[arg-type]
        graph=graph,
        top_k=3,
    )

    app = FastAPI()
    app.include_router(health.router)
    app.include_router(qa.router)
    app.dependency_overrides[get_state] = lambda: state
    return app


async def test_ask_streams_citation_tokens_and_done(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "What is on page 10?"})
        assert response.status_code == 200
        body = response.text

    assert "event: citation" in body
    assert body.count("event: token") >= 1
    assert "event: done" in body
    assert "page" in body  # at least one citation page made it through


async def test_health_endpoint_reports_count(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "collection": "test", "points": 3}
