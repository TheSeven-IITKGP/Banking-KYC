"""Run the ablation study.

  python eval/run_eval.py --retrieval-only          # free: retrieval metrics for all 5 configs
  python eval/run_eval.py                           # + generation (uses OpenAI credits)
  python eval/run_eval.py --judge                   # + LLM-judge faithfulness/correctness
  python eval/run_eval.py --configs 5_full --limit 10
  python eval/run_eval.py --agreement eval/results/<run>/manual_grading.csv

questions.csv columns: id, group, question, gold_answer, gold_paras, expect
  gold_paras = "CB-KYC-2025:28;CB-KYC-2025:65"   expect = answer | refuse
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kycrag.generate import format_context  # noqa: E402
from kycrag.guard import understand  # noqa: E402
from kycrag.retrieve import ABLATIONS, Retriever, order_for_context  # noqa: E402

JUDGE_PROMPT = """You grade a RAG answer about bank KYC rules.
Given CONTEXT, QUESTION, GOLD ANSWER and SYSTEM ANSWER, return JSON:
{"faithful": 1 if every factual claim in SYSTEM ANSWER is supported by CONTEXT else 0,
 "correct": 1 if SYSTEM ANSWER matches GOLD ANSWER in substance, 0.5 if partly, 0 if wrong or missing,
 "reason": "<one sentence>"}"""


def load_questions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question") and not r["id"].startswith("#")]
    for r in rows:
        r["gold"] = {tuple(g.split(":", 1)) for g in r.get("gold_paras", "").split(";") if ":" in g}
    return rows


def paras_of(hit_chunk) -> set[tuple[str, str]]:
    """(doc, para) pairs a chunk covers. Naive windows can cover several paragraphs."""
    return {(hit_chunk.doc_id, p) for p in hit_chunk.para_ref.split(",") if p}


def recall(gold: set, chunks) -> float | None:
    if not gold:
        return None
    covered = set().union(*(paras_of(c) for c in chunks)) if chunks else set()
    return len(gold & covered) / len(gold)


def run(args) -> None:
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    configs = [c for c in ABLATIONS if not args.configs or c.name in args.configs.split(",")]
    retrievers: dict[str, Retriever] = {}
    llm = judge = None
    if not args.retrieval_only:
        from kycrag.generate import LLMGenerator
        llm = LLMGenerator()
        if args.judge:
            from kycrag.config import SETTINGS
            judge = LLMGenerator(model=SETTINGS.judge_model)
    run_dir = ROOT / "eval" / "results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        if cfg.index not in retrievers:
            retrievers[cfg.index] = Retriever(cfg.index)
        r = retrievers[cfg.index]
        from kycrag.pipeline import KYCAssistant
        assistant = KYCAssistant(r, llm) if llm else None
        for q in questions:
            info = understand(q["question"], r.glossary)
            row = {"config": cfg.name, "id": q["id"], "group": q["group"], "expect": q["expect"]}
            if info.intent == "in_scope":
                hits = order_for_context(r.search(info.expanded, cfg, info.include_superseded))
            else:
                hits = []
            retrieved = [h.chunk for h in hits if h.via == "retrieved"]
            row["recall_at_5"] = recall(q["gold"], retrieved)
            row["context_recall"] = recall(q["gold"], [h.chunk for h in hits])
            row["refused_early"] = info.intent != "in_scope"
            row["refusal_correct"] = int(row["refused_early"] == (q["expect"] == "refuse"))  # overwritten below if generating
            if assistant:
                try:
                    res = assistant.ask(q["question"], cfg, log=False)
                except Exception as e:      # API error (rate limit, timeout, ...): skip, keep the run alive
                    row["status"] = "error"
                    row["judge_reason"] = f"generation error: {e}"
                    rows.append(row)
                    continue
                cited = [r.by_id[c] for c in res.get("citations", []) if c in r.by_id]
                if cited and q["gold"]:
                    row["citation_precision"] = sum(bool(paras_of(c) & q["gold"]) for c in cited) / len(cited)
                row["status"] = res["status"]
                declined = res["status"] in ("refused", "not_found", "unverified")
                row["refusal_correct"] = int(declined == (q["expect"] == "refuse"))
                row["answer"] = res["answer"]
                if judge and q["expect"] == "answer":
                    user = (f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {q['question']}\n\n"
                            f"GOLD ANSWER: {q['gold_answer']}\n\nSYSTEM ANSWER: {res['answer']}")
                    try:
                        g = json.loads(judge.complete(JUDGE_PROMPT, user))
                        row["faithful"], row["correct"], row["judge_reason"] = g.get("faithful"), g.get("correct"), g.get("reason", "")
                    except Exception as e:      # keep going; note the failure
                        row["judge_reason"] = f"judge error: {e}"
            rows.append(row)
        print(f"done {cfg.name}")
        write_outputs(run_dir, rows)   # write after every config so a later crash doesn't lose earlier progress


METRICS = ["recall_at_5", "context_recall", "citation_precision", "faithful", "correct", "refusal_correct"]


def write_outputs(run_dir: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "id", "group"), k))
    with (run_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = []
    for cfg in dict.fromkeys(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == cfg]
        line = {"config": cfg}
        for m in METRICS:
            vals = [float(r[m]) for r in sub if r.get(m) not in (None, "")]
            line[m] = round(sum(vals) / len(vals), 3) if vals else ""
        summary.append(line)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config"] + METRICS)
        w.writeheader()
        w.writerows(summary)
    md = ["| config | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    md += ["| " + " | ".join(str(s[k]) for k in ["config"] + METRICS) + " |" for s in summary]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    answered = [r for r in rows if r.get("answer") and r["config"] == "5_full"]
    if answered:
        sample = random.Random(7).sample(answered, min(20, len(answered)))
        with (run_dir / "manual_grading.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "answer", "correct", "human_correct"])
            w.writeheader()
            for r in sample:
                w.writerow({"id": r["id"], "answer": r["answer"], "correct": r.get("correct", ""), "human_correct": ""})
    print("\n".join(md))
    print(f"\nresults in {run_dir}")


def agreement(path: str) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correct"] != "" and r["correct"] != ""]
    if not rows:
        raise SystemExit("Fill the human_correct column first (0, 0.5 or 1).")
    same = sum(float(r["correct"]) == float(r["human_correct"]) for r in rows)
    print(f"judge-human exact agreement: {same}/{len(rows)} = {same / len(rows):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"))
    ap.add_argument("--configs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--agreement", default="")
    a = ap.parse_args()
    agreement(a.agreement) if a.agreement else run(a)
