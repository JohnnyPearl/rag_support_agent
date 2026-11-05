import re
from pathlib import Path

import pymupdf4llm

from app.schemas.chunks import ParsedBlock, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(.+?)\1")


def _clean_heading(title: str) -> str:
    while True:
        new = _MD_EMPHASIS_RE.sub(r"\2", title)
        if new == title:
            return title.strip()
        title = new


class PyMuPDFMarkdownParser:
    """Convert a PDF to Markdown with pymupdf4llm, then split each page into
    structural blocks (paragraphs / tables) annotated with a running section path."""

    def parse(self, path: Path) -> ParsedDocument:
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        blocks: list[ParsedBlock] = []
        section_stack: list[str] = []

        for page_data in pages:
            page_num = page_data["metadata"]["page_number"]
            md_text = page_data["text"]
            blocks.extend(_split_page(md_text, page_num, section_stack))

        return ParsedDocument(source=str(path), blocks=blocks)


def _split_page(md: str, page: int, section_stack: list[str]) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    buf: list[str] = []
    in_table = False

    def flush(is_table: bool) -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            blocks.append(
                ParsedBlock(
                    text=text,
                    page=page,
                    section_path=list(section_stack),
                    is_table=is_table,
                )
            )

    for line in md.splitlines():
        is_table_line = bool(_TABLE_LINE_RE.match(line))
        heading_match = _HEADING_RE.match(line)

        if is_table_line:
            if not in_table:
                flush(is_table=False)
                in_table = True
            buf.append(line)
            continue

        if in_table:
            flush(is_table=True)
            in_table = False

        if heading_match:
            flush(is_table=False)
            level = len(heading_match.group(1))
            title = _clean_heading(heading_match.group(2))
            del section_stack[level - 1:]
            section_stack.append(title)
            continue

        if line.strip() == "":
            if buf:
                flush(is_table=False)
            continue

        buf.append(line)

    flush(is_table=in_table)
    return blocks
