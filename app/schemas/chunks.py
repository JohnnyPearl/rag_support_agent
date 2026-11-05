from pydantic import BaseModel, Field


class ParsedBlock(BaseModel):
    """A single structural unit produced by the parser, before chunking."""

    text: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    is_table: bool = False


class ParsedDocument(BaseModel):
    source: str
    blocks: list[ParsedBlock]


class Chunk(BaseModel):
    chunk_id: str
    source: str = ""
    text: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    is_table: bool = False
    parent_section_id: str | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str = ""
    page: int
    section_path: list[str]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
