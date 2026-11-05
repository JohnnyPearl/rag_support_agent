"""Ingest a PDF into the local Qdrant store.

Usage:
    python -m scripts.ingest document/your_manual.pdf
"""
import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF document.")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    settings = get_settings()
    pipeline = IngestionPipeline(settings)

    print(f"ingesting {args.pdf} → {settings.qdrant_path}/{settings.collection_name}")
    stats = pipeline.run(args.pdf)
    print(
        f"done: parsed {stats['blocks']} blocks → "
        f"{stats['chunks']} chunks → {stats['stored']} points stored"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
