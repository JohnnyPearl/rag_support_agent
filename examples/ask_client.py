"""Minimal client for the /ask SSE endpoint.

Ingest a document and start the server first:
    python -m scripts.ingest document/your_manual.pdf
    uvicorn app.main:app --reload

Then ask one or more questions:
    python examples/ask_client.py "How do I install the system?" "What does the dashboard show?"

With no arguments, a couple of generic sample questions are asked.
"""
import argparse
import json
import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"


def ask(question: str) -> None:
    print(f"\n>>> {question}")
    with httpx.stream(
        "POST",
        f"{BASE_URL}/ask",
        json={"question": question},
        timeout=httpx.Timeout(60.0, read=None),
    ) as response:
        response.raise_for_status()
        event_name = "message"
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                _render(event_name, payload)
    print()


def _render(event: str, payload: dict) -> None:
    if event == "citation":
        print("Sources:")
        for c in payload["citations"]:
            section = " > ".join(c["section_path"]) or "(no section)"
            print(f"  - {c['source']} | page {c['page']:>3} | {section}")
        print("\nAnswer:")
    elif event == "token":
        sys.stdout.write(payload["text"])
        sys.stdout.flush()
    elif event == "done":
        return
    elif event == "error":
        print(f"\n[error] {payload['message']}")


SAMPLE_QUESTIONS = [
    "What can you help me with?",
    "Give me an overview of the system's main components.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions against the /ask SSE endpoint.")
    parser.add_argument("questions", nargs="*", help="Questions to ask (default: samples).")
    parser.add_argument("--url", default=BASE_URL, help=f"Server base URL (default: {BASE_URL}).")
    args = parser.parse_args()
    BASE_URL = args.url
    for q in args.questions or SAMPLE_QUESTIONS:
        ask(q)
