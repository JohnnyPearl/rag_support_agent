from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.schemas.chunks import Chunk, RetrievedChunk


class QdrantVectorStore:
    def __init__(self, path: Path, collection: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, items: list[tuple[Chunk, list[float]]]) -> None:
        points = [
            PointStruct(id=chunk.chunk_id, vector=vec, payload=chunk.model_dump())
            for chunk, vec in items
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit.payload), score=hit.score)
            for hit in result.points
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
