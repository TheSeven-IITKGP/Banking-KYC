"""Download every document in manifest.csv into data/raw and record date + checksum.

If a site blocks scripted downloads, save the page manually from your browser
(File > Save Page As > "Web Page, HTML only") under the `local_file` name and
run `python -m kycrag.cli checksum` to fill in the manifest.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from .config import SETTINGS
from .models import load_manifest, save_manifest

HEADERS = {"User-Agent": "Mozilla/5.0 (student capstone; KYC RAG research)"}


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def download_all(force: bool = False) -> None:
    import requests  # imported here so the rest of the package works without it

    SETTINGS.raw_dir.mkdir(parents=True, exist_ok=True)
    docs = load_manifest(SETTINGS.manifest_path)
    for d in docs:
        target = SETTINGS.raw_dir / d.local_file
        if target.exists() and not force:
            print(f"skip   {d.doc_id} (already downloaded)")
        else:
            print(f"fetch  {d.doc_id} <- {d.source_url}")
            r = requests.get(d.source_url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            target.write_bytes(r.content)
            d.downloaded_on = dt.date.today().isoformat()
        d.sha256 = sha256_of(target)
    save_manifest(SETTINGS.manifest_path, docs)


def fill_checksums() -> None:
    docs = load_manifest(SETTINGS.manifest_path)
    for d in docs:
        target = SETTINGS.raw_dir / d.local_file
        if target.exists():
            d.sha256 = sha256_of(target)
            d.downloaded_on = d.downloaded_on or dt.date.today().isoformat()
        else:
            print(f"missing {target}")
    save_manifest(SETTINGS.manifest_path, docs)
