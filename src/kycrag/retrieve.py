"""Hybrid retrieval with switchable components, so each ablation config is one flag set."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .bm25 import BM25
from .config import SETTINGS
from .embeddings import VectorStore, get_embedder
from .indexer import load_chunks
from .models import Chunk
from .timing import timer

@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    index: str = "paragraph"
    hybrid: bool = True
    rerank: bool = True
    xrefs: bool = True
    version_filter: bool = True

ABLATIONS = [
    RetrievalConfig("1_baseline", index="naive", hybrid=False, rerank=False, xrefs=False, version_filter=False),
    RetrievalConfig("2_paragraph_chunks", hybrid=False, rerank=False, xrefs=False, version_filter=False),
    RetrievalConfig("3_hybrid", hybrid=True, rerank=False, xrefs=False, version_filter=False),
    RetrievalConfig("4_rerank", hybrid=True, rerank=True, xrefs=False, version_filter=False),
    RetrievalConfig("5_full", hybrid=True, rerank=True, xrefs=True, version_filter=True),
]
FULL = ABLATIONS[-1]

AUTHORITY_ORDER = {("regulator", "current"): 0, ("internal", "current"): 2, ("regulator", "superseded"): 3}

@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str = "retrieved"

class Retriever:
    def __init__(self, index: str = "paragraph", embedder=None, reranker=None):
        folder = SETTINGS.processed_dir / index
        with timer(f"load_chunks[{index}]"):
            self.chunks = load_chunks(folder)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        with timer(f"load_vectors[{index}]"):
            self.store = VectorStore.load(folder)
        with timer("load_embedder"):
            self.embedder = embedder or get_embedder()
        if self.embedder.name != self.store.model_name:
            raise RuntimeError(f"Index built with {self.store.model_name}, but embedder is {self.embedder.name}. Rebuild the index.")
        with timer(f"build_bm25[{index}]"):
            self.bm25 = BM25([c.embed_text() for c in self.chunks])
        self._reranker = reranker
        gpath = SETTINGS.processed_dir / "glossary.json"
        self.glossary = json.loads(gpath.read_text()) if gpath.exists() else {}

    @property
    def reranker(self):
        if self._reranker is None:
            with timer("load_reranker"):
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(SETTINGS.rerank_model)
        return self._reranker

    def search(self, query: str, cfg: RetrievalConfig = FULL, include_superseded: bool = False) -> list[Hit]:
        with timer("search_total"):
            allowed = None
            if cfg.version_filter and not include_superseded:
                allowed = [c.status != "superseded" for c in self.chunks]

            with timer("embed_query"):
                qvec = self.embedder.embed_query(query)
            with timer("vector_search"):
                vec_hits = self.store.top_k(qvec, SETTINGS.vector_k, allowed)
            if cfg.hybrid:
                with timer("bm25_search"):
                    kw_hits = self.bm25.top_k(query, SETTINGS.bm25_k, allowed)
                with timer("rrf_fuse"):
                    fused = rrf([vec_hits, kw_hits], SETTINGS.rrf_k)
            else:
                fused = vec_hits

            candidates = [Hit(self.chunks[i], s) for i, s in fused]
            if cfg.rerank and candidates:
                with timer("rerank"):
                    pool = candidates[:20]
                    scores = self.reranker.predict([(query, h.chunk.embed_text()) for h in pool])
                    for h, s in zip(pool, scores):
                        h.score = float(s)
                    candidates = sorted(pool, key=lambda h: h.score, reverse=True)
            hits = candidates[:SETTINGS.final_k]

            if cfg.xrefs:
                with timer("xref_linking"):
                    hits += self._linked(hits, allowed)
        return hits

    def _linked(self, hits: list[Hit], allowed, max_extra: int = 4) -> list[Hit]:
        have = {h.chunk.chunk_id for h in hits}
        allowed_ids = None if allowed is None else {c.chunk_id for c, ok in zip(self.chunks, allowed) if ok}
        extra: list[Hit] = []
        for h in hits:
            for cid in h.chunk.cross_refs:
                if cid in have or (allowed_ids is not None and cid not in allowed_ids):
                    continue
                if cid in self.by_id and len(extra) < max_extra:
                    extra.append(Hit(self.by_id[cid], 0.0, via="linked"))
                    have.add(cid)
        return extra

def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: robust way to merge lists whose scores aren't comparable."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)

def order_for_context(hits: list[Hit]) -> list[Hit]:
    """Current regulation first, then FAQs/bank policy, then superseded text."""
    def key(h: Hit):
        c = h.chunk
        rank = AUTHORITY_ORDER.get((c.authority, c.status), 1)
        if c.doc_id.endswith("FAQ"):
            rank = 1
        return (rank, h.via != "retrieved")
    return sorted(hits, key=key)
