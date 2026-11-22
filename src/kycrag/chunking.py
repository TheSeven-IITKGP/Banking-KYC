"""Paragraph chunks (the real system) and fixed-size chunks (the ablation baseline)."""
from __future__ import annotations

import re

from .models import Chunk, DocMeta
from .parse import ParsedPara

CLAUSE_START_RE = re.compile(r"^\((\d{1,2}|[ivxl]{1,6}|[a-z])\)\s")
XREF_RE = re.compile(
    r"\bparagraphs?\s+((?:\d{1,3}(?:\s*\(\w{1,5}\))*\s*(?:,|and|or|to|&)?\s*)+)", re.I
)
# ‘Officially Valid Document (OVD)’   or   ‘Central KYC Records Registry’ (CKYCR)
DEF_INSIDE_RE = re.compile(r"[‘'\"]([^’'\"]{3,120}?)\s*\(([A-Z][A-Za-z\-]{1,10})\)[’'\"]")
DEF_OUTSIDE_RE = re.compile(r"[‘'\"]([^’'\"]{3,120})[’'\"]\s*\(([A-Z][A-Za-z\-]{1,10})\)")


def extract_xrefs(text: str, own_ref: str) -> list[str]:
    refs: list[str] = []
    for m in XREF_RE.finditer(text):
        span = re.sub(r"\(\w{1,5}\)", " ", m.group(1))     # drop clause numbers like (5)
        for num in re.findall(r"\d{1,3}", span):
            if num != own_ref and num not in refs:
                refs.append(num)
    return refs


def extract_glossary(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for rx in (DEF_INSIDE_RE, DEF_OUTSIDE_RE):
        for term, acr in rx.findall(text):
            out.setdefault(acr.strip(), term.strip())
    return out


def _split_long(lines: list[str], max_chars: int) -> list[list[str]]:
    """Split at clause boundaries so each part stays under max_chars."""
    parts: list[list[str]] = [[]]
    size = 0
    for ln in lines:
        if parts[-1] and CLAUSE_START_RE.match(ln) and size + len(ln) > max_chars:
            parts.append([])
            size = 0
        parts[-1].append(ln)
        size += len(ln) + 1
    return [p for p in parts if p]


def paragraph_chunks(paras: list[ParsedPara], doc: DocMeta, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paras:
        path_bits = [doc.title, p.chapter, p.section, f"Para {p.ref}" if p.ref[0].isdigit() else p.ref]
        heading_path = " > ".join(b for b in path_bits if b)
        is_def = "definition" in p.section.lower()
        xrefs = extract_xrefs(p.text, p.ref)
        groups = _split_long(p.lines, max_chars) if len(p.text) > max_chars else [p.lines]
        lead = p.lines[0][:300]
        for k, group in enumerate(groups):
            body = "\n".join(group)
            if k > 0:
                body = f"(continued) {lead}\n…\n{body}"
            cid = f"{doc.doc_id}:p{p.ref.replace(' ', '')}" + (f".{k}" if len(groups) > 1 else "")
            chunks.append(Chunk(
                chunk_id=cid, doc_id=doc.doc_id, para_ref=p.ref, heading_path=heading_path,
                text=body, status=doc.status, authority=doc.authority,
                version_date=doc.version_date, doc_title=doc.title, part=k,
                cross_refs=list(xrefs), is_definition=is_def,
            ))
    _resolve_xrefs(chunks)
    return chunks


def _resolve_xrefs(chunks: list[Chunk]) -> None:
    """Turn paragraph numbers into chunk ids within the same document."""
    by_para: dict[tuple[str, str], list[str]] = {}
    for c in chunks:
        by_para.setdefault((c.doc_id, c.para_ref), []).append(c.chunk_id)
    for c in chunks:
        ids: list[str] = []
        for ref in c.cross_refs:
            ids.extend(by_para.get((c.doc_id, ref), []))
        c.cross_refs = [i for i in ids if not i.startswith(f"{c.doc_id}:p{c.para_ref}")]


def naive_chunks(paras: list[ParsedPara], doc: DocMeta, size: int) -> list[Chunk]:
    """Baseline: concatenate the whole document and cut every `size` characters.
    para_ref records which paragraphs the window overlaps so recall can still be scored."""
    spans: list[tuple[int, int, str]] = []
    text, pos = "", 0
    for p in paras:
        t = p.text + "\n"
        spans.append((pos, pos + len(t), p.ref))
        text += t
        pos += len(t)
    chunks = []
    for k, start in enumerate(range(0, len(text), size)):
        end = start + size
        refs = [r for (s, e, r) in spans if s < end and e > start]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}:w{k}", doc_id=doc.doc_id, para_ref=",".join(refs),
            heading_path=doc.title, text=text[start:end], status=doc.status,
            authority=doc.authority, version_date=doc.version_date, doc_title=doc.title,
        ))
    return chunks
