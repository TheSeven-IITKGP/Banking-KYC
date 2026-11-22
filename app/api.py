"""FastAPI backend for the KYC assistant, with a static HTML/JS frontend.

Run:   uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000 in a browser, or call the JSON API directly:

  curl -s http://127.0.0.1:8000/api/ask \\
    -H "Content-Type: application/json" \\
    -d '{"question": "What are the limits on a small account?"}'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from kycrag.generate import LLMGenerator  # noqa: E402
from kycrag.pipeline import KYCAssistant  # noqa: E402
from kycrag.retrieve import ABLATIONS, Retriever  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
CONFIG_NAMES = [c.name for c in ABLATIONS]
DEFAULT_CONFIG = ABLATIONS[-1].name

app = FastAPI(title="KYC rules assistant")

_retrievers: dict[str, Retriever] = {}
_llm: LLMGenerator | None = None


class AskRequest(BaseModel):
    question: str
    config: str = DEFAULT_CONFIG


def _get_llm() -> LLMGenerator:
    global _llm
    if _llm is None:
        try:
            _llm = LLMGenerator()
        except KeyError as e:
            raise HTTPException(500, f"Missing environment variable {e}. Check your .env.") from e
    return _llm


def _get_retriever(index: str) -> Retriever:
    if index not in _retrievers:
        try:
            _retrievers[index] = Retriever(index)
        except FileNotFoundError as e:
            raise HTTPException(500, "No index found. Run `python -m kycrag.cli build` first.") from e
    return _retrievers[index]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/configs")
def configs() -> list[str]:
    return CONFIG_NAMES


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    cfg = next((c for c in ABLATIONS if c.name == req.config), None)
    if cfg is None:
        raise HTTPException(400, f"Unknown config {req.config!r}. Choose one of {CONFIG_NAMES}.")
    assistant = KYCAssistant(_get_retriever(cfg.index), _get_llm())
    return assistant.ask(req.question, cfg)


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
