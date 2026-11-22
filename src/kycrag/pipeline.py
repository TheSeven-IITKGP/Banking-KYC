"""ask(question) -> verified answer. Ties guard, retrieval, generation and checks together."""
from __future__ import annotations

import datetime as dt
import json

from .config import SETTINGS
from .generate import check_citations, cited_ids, generate_answer
from .guard import understand
from .retrieve import FULL, Hit, RetrievalConfig, Retriever, order_for_context
from .timing import timer

REFUSALS = {
    "out_of_scope": "This assistant only answers questions about bank KYC rules in its RBI and bank-policy corpus.",
    "customer_specific": ("This assistant answers questions about KYC rules, not about individual customers or "
                          "accounts. Please ask about the rule itself, without customer details."),
    "unverified": "I couldn't produce an answer with verifiable citations. Try rephrasing the question.",
}


class KYCAssistant:
    def __init__(self, retriever: Retriever, llm, corpus_version: str = ""):
        self.retriever = retriever
        self.llm = llm
        self.corpus_version = corpus_version or _corpus_version()

    def ask(self, question: str, cfg: RetrievalConfig = FULL, log: bool = True) -> dict:
        with timer(f"ask_total[{cfg.name}]"):
            info = understand(question, self.retriever.glossary)
            result: dict = {"question_redacted": info.redacted, "pii_found": info.pii_found,
                            "intent": info.intent, "config": cfg.name, "hits": [],
                            "corpus_version": self.corpus_version}
            if info.intent != "in_scope":
                result.update(status="refused", answer=REFUSALS[info.intent], citations=[], conflicts="")
                return self._finish(result, log)

            hits = order_for_context(self.retriever.search(info.expanded, cfg, info.include_superseded))
            result["hits"] = [_hit_view(h) for h in hits]
            with timer("generate"):
                out = generate_answer(self.llm, info.redacted, hits)
            with timer("check_citations"):
                bad = check_citations(out, hits)
            if bad:                                     # one retry with feedback
                with timer("generate_retry"):
                    out = generate_answer(self.llm, info.redacted, hits, feedback=f"These tags were not in the context: {bad}.")
                bad = check_citations(out, hits)
            if bad:
                result.update(status="unverified", answer=REFUSALS["unverified"], citations=[], conflicts="",
                              rejected_citations=bad)
            else:
                result.update(status=out.get("status", "found"), answer=out.get("answer", ""),
                              citations=cited_ids(out), conflicts=out.get("conflicts", ""))
            return self._finish(result, log)

    def _finish(self, result: dict, log: bool) -> dict:
        if log:
            SETTINGS.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {k: v for k, v in result.items() if k != "hits"}
            entry["hit_ids"] = [h["chunk_id"] for h in result["hits"]]
            entry["model"] = getattr(self.llm, "model", "unknown")
            entry["time"] = dt.datetime.now().isoformat(timespec="seconds")
            with SETTINGS.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")     # never logs the raw question
        return result


def _hit_view(h: Hit) -> dict:
    c = h.chunk
    return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "para_ref": c.para_ref, "heading_path": c.heading_path,
            "status": c.status, "authority": c.authority, "via": h.via, "score": round(h.score, 4), "text": c.text}


def _corpus_version() -> str:
    p = SETTINGS.processed_dir / "corpus.json"
    return json.loads(p.read_text())["corpus_version"] if p.exists() else "unbuilt"


def build_default_assistant() -> KYCAssistant:
    from .generate import LLMGenerator
    return KYCAssistant(Retriever("paragraph"), LLMGenerator())
