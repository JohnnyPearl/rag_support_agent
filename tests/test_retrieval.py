import random

from app.retrieval.vector_store.qdrant import QdrantVectorStore
from app.schemas.chunks import Chunk


def _vec(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def test_qdrant_upsert_and_search_round_trip(tmp_path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", collection="test")
    store.ensure_collection(dim=8)

    chunks = [
        Chunk(
            chunk_id=f"00000000-0000-5000-8000-00000000000{i}",
            text=f"chunk text {i}",
            page=i + 1,
            section_path=["Root", f"Sec{i}"],
            is_table=(i == 2),
        )
        for i in range(5)
    ]
    vectors = [_vec(8, seed=i) for i in range(5)]
    store.upsert(list(zip(chunks, vectors, strict=True)))

    assert store.count() == 5

    hits = store.search(vectors[2], top_k=3)
    assert len(hits) == 3
    top = hits[0].chunk
    assert top.chunk_id == chunks[2].chunk_id
    assert top.text == "chunk text 2"
    assert top.section_path == ["Root", "Sec2"]
    assert top.is_table is True
    assert top.page == 3


def test_qdrant_upsert_is_idempotent(tmp_path):
    store = QdrantVectorStore(path=tmp_path / "qdrant", collection="test")
    store.ensure_collection(dim=4)

    chunk = Chunk(
        chunk_id="00000000-0000-5000-8000-00000000000a",
        text="hello",
        page=1,
        section_path=["S"],
    )
    store.upsert([(chunk, _vec(4, 1))])
    store.upsert([(chunk, _vec(4, 1))])
    assert store.count() == 1
