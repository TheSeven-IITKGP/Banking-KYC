"""Central settings. Values come from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv(ROOT / ".env")

@dataclass
class Settings:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    manifest_path: Path = ROOT / "manifest.csv"
    log_path: Path = ROOT / "logs" / "queries.jsonl"

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", ""))

    embed_backend: str = field(default_factory=lambda: os.getenv("EMBED_BACKEND", "local"))
    local_embed_model: str = field(default_factory=lambda: os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    openai_embed_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"))

    bm25_k: int = 20
    vector_k: int = 20
    rrf_k: int = 60
    final_k: int = 5
    max_chunk_chars: int = 2500
    naive_chunk_chars: int = 2000

SETTINGS = Settings()
