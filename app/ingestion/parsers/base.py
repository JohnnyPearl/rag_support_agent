from pathlib import Path
from typing import Protocol

from app.schemas.chunks import ParsedDocument


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...
