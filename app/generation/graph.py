from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation.prompts import SYSTEM_PROMPT, build_user_message, format_context_block
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import DenseRetriever
from app.schemas.chunks import Citation, RetrievedChunk


class QueryState(TypedDict, total=False):
    question: str
    top_k: int
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    citations: list[Citation]
    system_prompt: str
    user_prompt: str


def build_graph(retriever: DenseRetriever, reranker: Reranker):
    """Compile the retrieve → rerank → prepare graph.

    Generation is deliberately NOT a node here so the FastAPI route can stream
    tokens directly from the LLM. When we add agentic loops (query rewriting,
    self-correction, etc.) generation moves into the graph and the route
    switches to `graph.astream_events`."""

    async def retrieve_node(state: QueryState) -> dict:
        hits = await retriever.retrieve(state["question"], state["top_k"])
        return {"retrieved": hits}

    async def rerank_node(state: QueryState) -> dict:
        ranked = await reranker.rerank(state["question"], state["retrieved"])
        return {"reranked": ranked}

    async def prepare_node(state: QueryState) -> dict:
        ranked = state["reranked"]
        context_blocks = [
            format_context_block(
                i + 1, r.chunk.source, r.chunk.page, r.chunk.section_path, r.chunk.text
            )
            for i, r in enumerate(ranked)
        ]
        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                source=r.chunk.source,
                page=r.chunk.page,
                section_path=r.chunk.section_path,
            )
            for r in ranked
        ]
        return {
            "citations": citations,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_user_message(state["question"], context_blocks),
        }

    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("prepare", prepare_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "prepare")
    graph.add_edge("prepare", END)
    return graph.compile()
