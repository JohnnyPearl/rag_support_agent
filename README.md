# VizMon RAG Support Agent: Documentation Q&A for HPC Monitoring

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://vizmon.vercel.app)
[![Paper DOI](https://img.shields.io/badge/DOI-10.3233%2FFAIA260739-blue)](https://doi.org/10.3233/FAIA260739)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

[Project Page](https://vizmon.vercel.app) · [Paper](https://doi.org/10.3233/FAIA260739) · [Visualization Agent](https://github.com/JohnnyPearl/text2sql2chart)

## Overview

This repository is a modular retrieval-augmented generation (RAG) service that answers questions about a PDF corpus, such as a system's user manual and technical reports. Answers stream over a Server-Sent Events (SSE) endpoint, and every answer comes with page-level citations naming the source document, page and section.

In VizMon, the agent lets operators ask how the monitoring system works: how to configure it, what a component does, or how to read a dashboard. It complements the [visualization agent](https://github.com/JohnnyPearl/text2sql2chart), which answers questions about the telemetry itself.

All ingested documents share a single Qdrant collection. Retrieval searches across all of them, and each citation names the document it came from.

## Pipeline

```
 Ingestion (offline, scripts/ingest.py)
 ──────────────────────────────────────
   PDF ──► pymupdf4llm parse ──► chunk ──► embed ──────────────────► Qdrant
          (Markdown + headings   (hierarchical,  (text-embedding-3-small,  (local,
           + page numbers)        token-sized)    1536-dim)                 file-backed)

 Question answering (online, POST /ask)
 ──────────────────────────────────────
   question ──► embed ──► retrieve (Qdrant, top-k) ──► rerank ──► prompt
                                                                    │
                                                                    ▼
   SSE: citation event + token events + done  ◄── gpt-4o-mini (streamed)
```

## Stack

- **API**: FastAPI + `sse-starlette`
- **Orchestration**: LangGraph state graph (retrieve → rerank → prepare prompt)
- **Vector DB**: Qdrant in local/embedded mode (file-backed, no daemon)
- **Parsing**: `pymupdf4llm` (PDF → Markdown with heading hierarchy + page numbers)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim)
- **LLM**: OpenAI `gpt-4o-mini`, streamed

Each pipeline stage sits behind a `Protocol` so it can be swapped without touching the rest.

## Project Structure

```
app/
├── main.py                # FastAPI app + lifespan
├── config.py              # pydantic-settings (env-driven)
├── api/routes/            # /health, /ask
├── ingestion/             # parse → chunk → embed → upsert
├── retrieval/             # embedder, vector store, retriever, reranker
├── generation/            # LLM, prompts, LangGraph
└── schemas/               # Chunk, Citation, request/response models

scripts/ingest.py          # one-shot CLI: PDF → Qdrant
examples/ask_client.py     # minimal SSE client for /ask
document/                  # put your PDFs here (gitignored)
data/qdrant/               # local Qdrant store (gitignored)
tests/                     # chunking, retrieval, /ask route
```

## Quick Start

### 1. Install

Requires Python 3.11+. The project is installed as an editable package so `python -m scripts.ingest` and `uvicorn app.main:app` resolve.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY
```

All settings are read from environment variables (or `.env`) via pydantic-settings. See [.env.example](.env.example) for the full list and defaults.

### 3. Ingest your documents

Put your PDFs in [document/](document/), then run the ingest script once per PDF. All documents share the same collection.

```bash
python -m scripts.ingest document/your_manual.pdf
python -m scripts.ingest document/your_technical_report.pdf
# each prints: parsed N blocks → N chunks → N points stored (cumulative)
```

Re-runs are idempotent per source file (chunks have stable UUIDv5 IDs). To rebuild from scratch, delete the store with `rm -rf data/qdrant` first.

### 4. Run the server

```bash
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
```

The server refuses to start if the Qdrant collection doesn't exist, so ingest first.

### 5. Ask a question

```bash
curl -N -X POST http://127.0.0.1:8000/ask \
    -H 'Content-Type: application/json' \
    -d '{"question": "How do I add a new node to the monitoring system?"}'
```

Or use the included client, which prints citations followed by the streamed answer:

```bash
python examples/ask_client.py "How do I add a new node to the monitoring system?"
```

## API Reference

### `POST /ask`

Request body: `{"question": "..."}`. The response is a stream of Server-Sent Events:

| event      | payload                              |
|------------|--------------------------------------|
| `citation` | list of `{chunk_id, source, page, section_path}` for the chunks the LLM was given |
| `token`    | one LLM token at a time              |
| `done`     | end of stream                        |
| `error`    | error message (terminal)             |

### `GET /health`

Returns `{"status": "ok", "collection": "<name>", "points": <count>}`.

## Tests

```bash
pytest
```

Covers chunker shapes, the Qdrant upsert/search round-trip, and the `/ask` SSE route (with the embedder and LLM mocked). No API key or network access is needed.

## Where to extend

The seams are the `Protocol` classes:

| To add…                        | Implement…                                                  |
|--------------------------------|-------------------------------------------------------------|
| Hybrid (BM25 + dense) search   | a new retriever class implementing `DenseRetriever`'s shape |
| Reranking                      | replace `PassthroughReranker` in `app/retrieval/reranker.py` |
| Other source formats           | new parser implementing `DocumentParser`                    |
| Agentic loops (rewrite, retry) | add nodes/edges in `app/generation/graph.py` and switch the route to `graph.astream_events` |

## Citation

This agent is part of the VizMon system described in the paper below. If you use this code, please cite:

```bibtex
@inproceedings{dang2026vizmon,
  title     = {A Smart Monitoring Architecture for {HPC} Clusters with an {AI}-Powered Natural Language Visualization Agent},
  author    = {Dang, Ngoc-Phu and Le, Khanh and Nguyen, Viet-Hung and Pham, Thai-An and La-Quoc, Nhut-Huan and Thoai, Nam},
  booktitle = {New Trends in Intelligent Software Methodologies, Tools and Techniques (SoMeT 2026)},
  editor    = {Fujita, Hamido and Selamat, Ali and Mohamed, Farhan},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {430},
  pages     = {193--202},
  publisher = {IOS Press},
  year      = {2026},
  doi       = {10.3233/FAIA260739}
}
```

## Acknowledgments

This work was carried out at the High Performance Computing Laboratory (HPCC), Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, under the supervision of Assoc. Prof. Nam Thoai and Dr. Thanh-Dang Diep.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
