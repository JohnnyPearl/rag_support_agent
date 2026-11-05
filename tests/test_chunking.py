from app.ingestion.chunking.hierarchical import HierarchicalChunker
from app.ingestion.parsers.pymupdf_md import _clean_heading, _split_page
from app.schemas.chunks import ParsedBlock, ParsedDocument


def test_split_page_tracks_heading_hierarchy():
    md = (
        "# Chapter 1\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Section 1.1\n"
        "\n"
        "Body of section 1.1.\n"
        "\n"
        "## Section 1.2\n"
        "\n"
        "Body of section 1.2.\n"
    )
    stack: list[str] = []
    blocks = _split_page(md, page=7, section_stack=stack)
    assert [b.section_path for b in blocks] == [
        ["Chapter 1"],
        ["Chapter 1", "Section 1.1"],
        ["Chapter 1", "Section 1.2"],
    ]
    assert all(b.page == 7 for b in blocks)
    assert stack == ["Chapter 1", "Section 1.2"]  # mutation persists across pages


def test_split_page_groups_table_rows_into_one_block():
    md = (
        "Some intro.\n"
        "\n"
        "| col a | col b |\n"
        "|---|---|\n"
        "| x | y |\n"
        "| u | v |\n"
        "\n"
        "Trailing paragraph.\n"
    )
    blocks = _split_page(md, page=1, section_stack=[])
    table_blocks = [b for b in blocks if b.is_table]
    assert len(table_blocks) == 1
    assert "col a" in table_blocks[0].text
    assert "u | v" in table_blocks[0].text
    assert sum(1 for b in blocks if not b.is_table) == 2


def test_clean_heading_strips_emphasis_markers():
    assert _clean_heading("**REPORT**") == "REPORT"
    assert _clean_heading("_6.1 Intro_") == "6.1 Intro"
    assert _clean_heading("**Bold _and_ italic**") == "Bold and italic"


def test_chunker_keeps_short_block_intact():
    chunker = HierarchicalChunker(target_tokens=200, overlap_tokens=20)
    doc = ParsedDocument(
        source="manual.pdf",
        blocks=[ParsedBlock(text="short paragraph", page=1, section_path=["Intro"])],
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short paragraph"
    assert chunks[0].page == 1
    assert chunks[0].section_path == ["Intro"]
    assert chunks[0].parent_section_id is not None


def test_chunker_splits_oversized_text_block_with_overlap():
    chunker = HierarchicalChunker(target_tokens=20, overlap_tokens=4)
    big_text = " ".join(f"word{i}" for i in range(200))
    doc = ParsedDocument(
        source="manual.pdf",
        blocks=[ParsedBlock(text=big_text, page=2, section_path=["A"])],
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) > 5
    ids = {c.chunk_id for c in chunks}
    assert len(ids) == len(chunks)


def test_chunker_never_splits_table_blocks():
    chunker = HierarchicalChunker(target_tokens=10, overlap_tokens=2)
    big_table = "| a | b |\n" + "\n".join(f"| {i} | {i+1} |" for i in range(50))
    doc = ParsedDocument(
        source="manual.pdf",
        blocks=[ParsedBlock(text=big_table, page=3, section_path=["T"], is_table=True)],
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].is_table
    assert chunks[0].text == big_table


def test_chunk_ids_are_deterministic():
    doc = ParsedDocument(
        source="manual.pdf",
        blocks=[ParsedBlock(text="hello world", page=1, section_path=["S"])],
    )
    a = HierarchicalChunker(target_tokens=200, overlap_tokens=20).chunk(doc)
    b = HierarchicalChunker(target_tokens=200, overlap_tokens=20).chunk(doc)
    assert a[0].chunk_id == b[0].chunk_id
