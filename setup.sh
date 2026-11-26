#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
[ -f .env ] || cp .env.example .env
python -m kycrag.cli download
python -m kycrag.cli build
python -m kycrag.cli inspect CB-KYC-2025
echo
echo "Done. Next: put your free Groq key (GROQ_API_KEY) in .env, then run:  streamlit run app/streamlit_app.py"
