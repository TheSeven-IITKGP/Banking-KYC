# KYC rules RAG (capstone)

Answers questions about Indian bank KYC rules from RBI Directions, citing the exact paragraph for every claim. Built for an ablation study: each retrieval component can be switched on or off and measured.

Study aid over public regulatory text. Not compliance or legal advice.

## Quick start

Unzip into `Desktop/ai`, open a terminal in `Desktop/ai/kyc-rag`, and run `bash setup.sh` (macOS/Linux) or `setup.bat` (Windows). It installs everything, downloads the documents, and builds the index.

## Setup (manual)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # then put your free Groq key in .env (never commit it)
```

## Run

```bash
python -m kycrag.cli download          # or save pages manually into data/raw (see manifest.csv local_file)
python -m kycrag.cli checksum          # only if you saved files manually
python -m kycrag.cli build             # parse, chunk, embed (local model, free)
python -m kycrag.cli inspect CB-KYC-2025   # CHECK THIS: paragraph count, gaps, merged paragraphs
python -m kycrag.cli ask "What are the limits on a small account?"
streamlit run app/streamlit_app.py     # opens on http://127.0.0.1:8501 only
```

For the 2025 Commercial Banks Directions (updated as on Oct 1, 2026), `inspect` should report paragraphs numbered from 1 with no gaps. Gaps or a few huge paragraphs mean the parser needs adjusting for your downloaded copy.

## Evaluate

```bash
python eval/run_eval.py --retrieval-only     # free: recall for all 5 configs
python eval/run_eval.py --configs 5_full     # generation (uses credits)
python eval/run_eval.py --judge              # + LLM judge (uses more credits)
python eval/run_eval.py --agreement eval/results/<run>/manual_grading.csv
```

Configs: `1_baseline` (fixed 2,000-char windows, vectors only) → `2_paragraph_chunks` → `3_hybrid` (+BM25, RRF) → `4_rerank` (+cross-encoder) → `5_full` (+cross-references, version filter). Results land in `eval/results/<timestamp>/summary.md`.

`eval/questions.csv` has 14 starter questions. Grow it to 80–100 and verify every gold paragraph number against your downloaded copy.

## Tests

```bash
python -m unittest discover -s tests -v      # 21 tests, fully offline (no model, no API)
```

## Layout

| Path | What it does |
|---|---|
| `src/kycrag/parse.py` | HTML/PDF → paragraphs with chapter and section |
| `src/kycrag/chunking.py` | Paragraph chunks, heading paths, cross-references, glossary; naive baseline chunks |
| `src/kycrag/bm25.py` | Keyword search (no dependency) |
| `src/kycrag/embeddings.py` | Local / OpenAI / test embedders; numpy vector store |
| `src/kycrag/indexer.py` | Builds both indexes and the corpus version ID |
| `src/kycrag/guard.py` | PII redaction, intent (in scope / customer-specific / off-topic), acronym expansion |
| `src/kycrag/retrieve.py` | Hybrid retrieval, rerank, cross-ref expansion, version filter, ablation configs |
| `src/kycrag/generate.py` | Groq/OpenAI JSON answer with inline tags; citation check |
| `src/kycrag/pipeline.py` | `ask()` end to end, one retry on bad citations, redacted logging |
| `app/streamlit_app.py` | Demo UI with a config switcher |
| `eval/run_eval.py` | Ablation runner, metrics, judge, judge–human agreement |

## Privacy and safety

Questions are redacted (Aadhaar, PAN, phone, email, account/card numbers, IFSC) before retrieval, before anything is sent to the LLM provider (Groq by default), and before logging. The raw question is never written to disk. Streamlit binds to 127.0.0.1 only.

## Open questions and known limits

See the project doc, section "Open questions and blockers". In short: the parser, model downloads, reranker, OpenAI calls, PDF parsing and the UI were written without network access and are untested against real files until you run them.
