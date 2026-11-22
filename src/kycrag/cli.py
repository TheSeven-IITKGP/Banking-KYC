"""Usage:
  python -m kycrag.cli download            # fetch documents listed in manifest.csv
  python -m kycrag.cli checksum            # after saving files manually
  python -m kycrag.cli build [--embed local|openai|hash]
  python -m kycrag.cli inspect CB-KYC-2025 # check paragraph parsing for one doc
  python -m kycrag.cli ask "What are the limits on a small account?"
"""
from __future__ import annotations

import argparse
import json

from .config import SETTINGS


def main() -> None:
    ap = argparse.ArgumentParser(prog="kycrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    sub.add_parser("checksum")
    b = sub.add_parser("build")
    b.add_argument("--embed", choices=["local", "openai", "hash"], default=None)
    i = sub.add_parser("inspect")
    i.add_argument("doc_id")
    a = sub.add_parser("ask")
    a.add_argument("question")
    args = ap.parse_args()

    if args.cmd == "download":
        from .download import download_all
        download_all(force=args.force)
    elif args.cmd == "checksum":
        from .download import fill_checksums
        fill_checksums()
    elif args.cmd == "build":
        from .indexer import build_all
        build_all(args.embed)
    elif args.cmd == "inspect":
        path = SETTINGS.processed_dir / f"{args.doc_id}.paras.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        refs = [r["ref"] for r in rows]
        nums = [int(r) for r in refs if r.isdigit()]
        missing = sorted(set(range(1, max(nums) + 1)) - set(nums)) if nums else []
        print(f"{len(rows)} paragraphs; numbered 1..{max(nums) if nums else 0}; missing numbers: {missing or 'none'}")
        big = sorted(rows, key=lambda r: -r["chars"])[:5]
        print("largest paragraphs (check these are not merged):")
        for r in big:
            print(f"  para {r['ref']:>4}  {r['chars']:>6} chars  {r['section'][:40]!r}  {r['start'][:60]!r}")
    elif args.cmd == "ask":
        from .pipeline import build_default_assistant
        res = build_default_assistant().ask(args.question)
        print(json.dumps({k: v for k, v in res.items() if k != "hits"}, indent=2, ensure_ascii=False))
        print("\nRetrieved:", ", ".join(h["chunk_id"] for h in res["hits"]))


if __name__ == "__main__":
    main()
