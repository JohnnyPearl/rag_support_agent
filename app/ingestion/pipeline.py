from pathlib import Path

from app.config import Settings
from app.ingestion.chunking.hierarchical import HierarchicalChunker
from app.ingestion.parsers.pymupdf_md import PyMuPDFMarkdownParser
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.vector_store.qdrant import QdrantVectorStore


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.parser = PyMuPDFMarkdownParser()
        self.chunker = HierarchicalChunker(
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
        self.store = QdrantVectorStore(
            path=settings.qdrant_path,
            collection=settings.collection_name,
        )
        self.embedding_dim = settings.embedding_dim

    def run(self, pdf_path: Path) -> dict[str, int]:
        doc = self.parser.parse(pdf_path)
        chunks = self.chunker.chunk(doc)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.ensure_collection(self.embedding_dim)
        self.store.upsert(list(zip(chunks, vectors, strict=True)))
        return {
            "blocks": len(doc.blocks),
            "chunks": len(chunks),
            "stored": self.store.count(),
        }
