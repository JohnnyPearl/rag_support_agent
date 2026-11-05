import uuid
from pathlib import Path

import tiktoken

from app.schemas.chunks import Chunk, ParsedBlock, ParsedDocument

_NAMESPACE = uuid.UUID("a8f5f167-2b6a-5a4d-9f3e-1c8d7b6a5e4f")


class HierarchicalChunker:
    """Block-aware chunker. Tables stay whole. Long text blocks are split into
    overlapping token windows. Section path is preserved on every chunk."""

    def __init__(self, target_tokens: int, overlap_tokens: int, encoding: str = "cl100k_base"):
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.enc = tiktoken.get_encoding(encoding)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for ordinal, block in enumerate(doc.blocks):
            if block.is_table:
                chunks.append(self._make(doc.source, block.text, block, ordinal, 0))
                continue

            tokens = self.enc.encode(block.text)
            if len(tokens) <= self.target:
                chunks.append(self._make(doc.source, block.text, block, ordinal, 0))
                continue

            step = self.target - self.overlap
            for sub, start in enumerate(range(0, len(tokens), step)):
                window = tokens[start:start + self.target]
                if not window:
                    break
                chunks.append(self._make(doc.source, self.enc.decode(window), block, ordinal, sub))
                if start + self.target >= len(tokens):
                    break
        return chunks

    def _make(
        self, source: str, text: str, block: ParsedBlock, ordinal: int, sub: int
    ) -> Chunk:
        section_path = block.section_path
        parent = _section_id(source, section_path) if section_path else None
        return Chunk(
            chunk_id=_chunk_id(source, section_path, ordinal, sub),
            source=Path(source).name,
            text=text,
            page=block.page,
            section_path=section_path,
            is_table=block.is_table,
            parent_section_id=parent,
        )


def _chunk_id(source: str, section_path: list[str], ordinal: int, sub: int) -> str:
    name = f"{source}|{'/'.join(section_path)}|{ordinal}|{sub}"
    return str(uuid.uuid5(_NAMESPACE, name))


def _section_id(source: str, section_path: list[str]) -> str:
    name = f"{source}|section|{'/'.join(section_path)}"
    return str(uuid.uuid5(_NAMESPACE, name))
