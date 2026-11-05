from typing import Protocol

from app.schemas.chunks import Chunk, ParsedDocument


class Chunker(Protocol):
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...
