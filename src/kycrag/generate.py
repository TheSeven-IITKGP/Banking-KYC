"""Grounded answer generation (OpenAI) and the citation check."""
from __future__ import annotations

import json
import re

from .config import SETTINGS
from .retrieve import Hit
from .timing import timer

SYSTEM_PROMPT = """You answer questions about Indian bank KYC rules using ONLY the context passages provided.

Rules:
1. Use only facts stated in the context. Never use outside knowledge.
2. After every sentence that states a rule, add the tag of the passage it comes from, exactly as written, e.g. [CB-KYC-2025:p28].
3. Prefer passages marked CURRENT. If you use a SUPERSEDED passage, say it is superseded and state what the current rule is, if the context gives it.
4. If an INTERNAL bank policy passage disagrees with a regulator passage, say so explicitly and follow the regulator.
5. If the context does not contain the answer, set status to "not_found" and say so. Do not guess.
6. Describe what the rule says; do not give compliance or legal advice.

Reply with a JSON object only:
{"status": "found" | "partial" | "not_found",
 "answer": "<answer with inline [tag] citations>",
 "citations": ["<tag>", ...],
 "conflicts": "<one sentence on any conflict between sources, or empty>"}"""

TAG_RE = re.compile(r"\[([A-Za-z0-9\-]+:p[\w.]+)\]")


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        c = h.chunk
        label = "SUPERSEDED" if c.status == "superseded" else "CURRENT"
        kind = "INTERNAL bank policy" if c.authority == "internal" else "RBI"
        via = " (linked by cross-reference)" if h.via == "linked" else ""
        blocks.append(f"[{c.chunk_id}] {label} | {kind} | {c.heading_path}{via}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


class LLMGenerator:
    """Chat-completions generator. Groq and OpenAI both speak the OpenAI API shape,
    so the same client just points at a different base_url."""

    def __init__(self, model: str | None = None):
        import os

        from openai import OpenAI

        if SETTINGS.llm_provider == "groq":
            self.client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
            self.model = model or SETTINGS.groq_model
        else:
            self.client = OpenAI()      # reads OPENAI_API_KEY from the environment
            self.model = model or SETTINGS.openai_model

    def complete(self, system: str, user: str) -> str:
        with timer(f"llm_call[{self.model}]"):
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        return resp.choices[0].message.content or "{}"


def generate_answer(llm, question: str, hits: list[Hit], feedback: str = "") -> dict:
    user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    if feedback:
        user += f"\n\nYour previous answer failed a check: {feedback} Use only tags that appear in the context."
    raw = llm.complete(SYSTEM_PROMPT, user)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {"status": "not_found", "answer": raw, "citations": [], "conflicts": ""}
    out.setdefault("citations", [])
    out.setdefault("conflicts", "")
    return out


def check_citations(result: dict, hits: list[Hit]) -> list[str]:
    """Return citation tags that were NOT in the retrieved context (should be empty)."""
    allowed = {h.chunk.chunk_id for h in hits}
    cited = set(result.get("citations", [])) | set(TAG_RE.findall(result.get("answer", "")))
    return sorted(c for c in cited if c not in allowed)


def cited_ids(result: dict) -> list[str]:
    seen: list[str] = []
    for c in list(result.get("citations", [])) + TAG_RE.findall(result.get("answer", "")):
        if c not in seen:
            seen.append(c)
    return seen
