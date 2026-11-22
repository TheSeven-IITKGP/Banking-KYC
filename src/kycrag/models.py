"""Shared data structures."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DocMeta:
    doc_id: str
    title: str
    source_url: str
    issuer: str
    doc_type: str          # direction | amendment | faq | bank_policy
    entity_type: str       # commercial_bank | small_finance_bank | ...
    version_date: str      # YYYY-MM-DD, the "updated as on" date
    status: str            # current | superseded
    authority: str         # regulator | internal
    local_file: str        # file name inside data/raw
    parser: str            # rbi_html | rbi_pdf | bank_pdf | faq_html | faq_pdf
    downloaded_on: str = ""
    sha256: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    para_ref: str              # e.g. "28" (paragraph number as cited)
    heading_path: str          # "Doc title > Chapter VI ... > A. ... > Para 28"
    text: str                  # paragraph text (or part of it)
    status: str
    authority: str
    version_date: str
    doc_title: str
    part: int = 0              # >0 when a long paragraph was split
    cross_refs: list[str] = field(default_factory=list)   # chunk_ids this chunk refers to
    is_definition: bool = False

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}, para {self.para_ref}]"

    def embed_text(self) -> str:
        """Text used for embedding and BM25: heading path gives context to bare clauses."""
        return f"{self.heading_path}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def load_manifest(path: Path) -> list[DocMeta]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("doc_id") and not r["doc_id"].startswith("#")]
    return [DocMeta(**{k: (v or "").strip() for k, v in r.items()}) for r in rows]


def save_manifest(path: Path, docs: list[DocMeta]) -> None:
    fields = list(DocMeta.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in docs:
            w.writerow(asdict(d))
