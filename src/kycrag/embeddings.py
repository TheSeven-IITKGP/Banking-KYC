"""Embedding backends + a tiny numpy vector store.

A corpus of a few thousand chunks fits in memory, so cosine search over a numpy
matrix is exact, fast and dependency-free (no ChromaDB server or files to manage).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SETTINGS

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Free, CPU-friendly. First run downloads the model (~130 MB for bge-small)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or SETTINGS.local_embed_model
        self.model = SentenceTransformer(self.name)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0]


class OpenAIEmbedder:
    """Uses API credits. Only the public corpus text and (redacted) queries are sent."""

    def __init__(self, model_name: str | None = None):
        from openai import OpenAI

        self.name = model_name or SETTINGS.openai_embed_model
        self.client = OpenAI()

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 100):
            resp = self.client.embeddings.create(model=self.name, input=texts[i:i + 100])
            out.extend(d.embedding for d in resp.data)
        arr = np.array(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class HashEmbedder:
    """Deterministic bag-of-words hashing. For tests only: no model, no network."""

    name = "hash-test"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        from .bm25 import tokenize

        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def get_embedder(backend: str | None = None):
    backend = backend or SETTINGS.embed_backend
    if backend == "local":
        return LocalEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBED_BACKEND {backend!r}")


class VectorStore:
    def __init__(self, matrix: np.ndarray, model_name: str):
        self.matrix = matrix.astype(np.float32)
        self.model_name = model_name

    def top_k(self, qvec: np.ndarray, k: int, allowed: list[bool] | None = None) -> list[tuple[int, float]]:
        sims = self.matrix @ qvec.astype(np.float32)
        if allowed is not None:
            sims = np.where(np.array(allowed), sims, -np.inf)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx if np.isfinite(sims[i])]

    def save(self, folder: Path) -> None:
        np.save(folder / "embeddings.npy", self.matrix)
        (folder / "embeddings.meta.json").write_text(json.dumps({"model": self.model_name}))

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        meta = json.loads((folder / "embeddings.meta.json").read_text())
        return cls(np.load(folder / "embeddings.npy"), meta["model"])
