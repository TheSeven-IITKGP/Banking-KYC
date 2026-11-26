"""Build the two indexes: `paragraph` (real system) and `naive` (ablation baseline)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from .chunking import extract_glossary, naive_chunks, paragraph_chunks
from .config import SETTINGS
from .embeddings import VectorStore, get_embedder
from .models import Chunk, load_manifest
from .parse import parse_file

BASE_GLOSSARY = {
    "KYC": "Know Your Customer", "CDD": "Customer Due Diligence", "EDD": "Enhanced Due Diligence",
    "AML": "Anti-Money Laundering", "CFT": "Countering the Financing of Terrorism",
    "PEP": "Politically Exposed Person", "BO": "Beneficial Owner", "UBO": "Ultimate Beneficial Owner",
    "OVD": "Officially Valid Document", "CKYCR": "Central KYC Records Registry",
    "UCIC": "Unique Customer Identification Code", "V-CIP": "Video based Customer Identification Process",
    "VCIP": "Video based Customer Identification Process", "CIP": "Customer Identification Procedure",
    "STR": "Suspicious Transaction Report", "CTR": "Cash Transaction Report",
    "PAN": "Permanent Account Number", "NRI": "Non-Resident Indian", "PIO": "Person of Indian Origin",
    "FPI": "Foreign Portfolio Investor", "SHG": "Self Help Group", "LE": "Legal Entity",
    "PML": "Prevention of Money-Laundering", "PMLA": "Prevention of Money-Laundering Act",
    "FIU": "Financial Intelligence Unit", "NPO": "Non-profit organisation", "RE": "Regulated Entity",
}

def build_all(embed_backend: str | None = None) -> dict:
    docs = load_manifest(SETTINGS.manifest_path)
    out = SETTINGS.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    para_chunks: list[Chunk] = []
    base_chunks: list[Chunk] = []
    glossary = dict(BASE_GLOSSARY)
    report = {}
    for d in docs:
        path = SETTINGS.raw_dir / d.local_file
        if not path.exists():
            print(f"!! missing {path} — run `python -m kycrag.cli download` or save it manually")
            continue
        paras = parse_file(path, d.parser)
        with (out / f"{d.doc_id}.paras.jsonl").open("w", encoding="utf-8") as f:
            for p in paras:
                f.write(json.dumps({"ref": p.ref, "chapter": p.chapter, "section": p.section,
                                    "chars": len(p.text), "start": p.text[:120]}, ensure_ascii=False) + "\n")
        pcs = paragraph_chunks(paras, d, SETTINGS.max_chunk_chars)
        para_chunks += pcs
        base_chunks += naive_chunks(paras, d, SETTINGS.naive_chunk_chars)
        for c in pcs:
            if c.is_definition:
                for acr, term in extract_glossary(c.text).items():
                    glossary.setdefault(acr, term)
        report[d.doc_id] = {"paragraphs": len(paras), "chunks": len(pcs)}
        print(f"parsed {d.doc_id}: {len(paras)} paragraphs -> {len(pcs)} chunks")

    if not para_chunks:
        raise SystemExit("No documents parsed. Put files in data/raw first.")

    embedder = get_embedder(embed_backend)
    version = _corpus_version(docs)
    for name, chunks in (("paragraph", para_chunks), ("naive", base_chunks)):
        folder = out / name
        folder.mkdir(exist_ok=True)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        texts = [c.embed_text() for c in chunks]
        VectorStore(embedder.embed_docs(texts), embedder.name).save(folder)
    (out / "glossary.json").write_text(json.dumps(glossary, indent=1, ensure_ascii=False))
    meta = {"corpus_version": version, "built_on": dt.datetime.now().isoformat(timespec="seconds"),
            "embed_model": embedder.name, "docs": report}
    (out / "corpus.json").write_text(json.dumps(meta, indent=1))
    print(f"corpus_version {version}")
    return meta

def _corpus_version(docs) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(f"{d.doc_id}|{d.sha256}|{d.status}|{d.version_date}".encode())
    h.update(f"{SETTINGS.max_chunk_chars}|{SETTINGS.naive_chunk_chars}".encode())
    return h.hexdigest()[:12]

def load_chunks(folder: Path) -> list[Chunk]:
    with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f]
