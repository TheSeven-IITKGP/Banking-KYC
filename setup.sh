#!/usr/bin/env bash
# One-step setup for macOS / Linux. Run from inside the kyc-rag folder:  bash setup.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
[ -f .env ] || cp .env.example .env
python -m kycrag.cli download        # fetches the 4 public documents into data/raw
python -m kycrag.cli build           # parses, chunks and embeds (downloads the embedding model once)
python -m kycrag.cli inspect CB-KYC-2025
echo
echo "Done. Next: put your free Groq key (GROQ_API_KEY) in .env, then run:  streamlit run app/streamlit_app.py"
