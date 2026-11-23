# FinKYC: Unified Enterprise Small-Model Adaptation & RBI Regulatory RAG Platform

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Track](https://img.shields.io/badge/Capstone-Track%20A%20Model%20Side-purple.svg)]()
[![Lead Engineer](https://img.shields.io/badge/Author-Chandan%20Bauri-orange.svg)](mailto:chandanbauri441@gmail.com)

**FinKYC** is a unified, production-grade enterprise platform combining:
1. **Model-Side Adaptation (Track A)**: Adapts small open 7–8B models (Llama-3, Qwen-2.5, Mistral) via QLoRA for noisy OCR document extraction and verification across Indian identity documents (Aadhaar, PAN, Voter ID, Driving Licence, Passport).
2. **RBI Regulatory RAG Engine**: Retrieval-Augmented Generation over RBI Master Directions on KYC, delivering paragraph-level citations, hybrid BM25 + dense retrieval, cross-reference expansion, and compliance ablation benchmarking.

---

## Unified System Architecture

```text
Banking-KYC/
├── banking_kyc/               # Track A: Model-Side Fine-Tuning, OCR & Security
│   ├── api/                   # REST API server
│   ├── config.py              # Dynamic thresholds and settings
│   ├── finetuning/            # QLoRA configs, dataset splits, evaluator
│   ├── models.py              # Pydantic domain models
│   ├── ocr/                   # Document normalizer, quality analyzer, heuristics
│   ├── providers/             # Pluggable multi-provider engine (vLLM, Ollama, HF, Local)
│   ├── security/              # Prompt injection guard & PII redactor
│   ├── web/                   # User Portal & Admin Control Center (HTML5/CSS3/JS)
│   └── workflow/              # LangGraph KYC verification StateGraph
├── src/kycrag/                # Regulatory RAG Subsystem
│   ├── bm25.py                # Zero-dependency BM25 keyword search
│   ├── chunking.py            # Hierarchical paragraph chunking & xrefs
│   ├── embeddings.py          # Dense embeddings & numpy vector index
│   ├── guard.py               # PII redaction & intent classification
│   ├── indexer.py             # Dual-index builder & version tracking
│   ├── parse.py               # RBI Directions HTML/PDF parser
│   ├── pipeline.py            # End-to-end RAG query pipeline
│   └── retrieve.py            # Hybrid RRF retrieval & ablation runner
├── app/                       # RAG Web Interfaces
│   ├── api.py                 # FastAPI RAG endpoint
│   ├── streamlit_app.py       # Streamlit RAG exploration UI
│   └── static/index.html      # RAG web view
├── eval/                      # RAG Evaluation Suite
│   ├── questions.csv          # Benchmark evaluation questions
│   └── run_eval.py            # RAG ablation runner & LLM judge
├── tests/                     # Unified Test Suite (36+ tests)
│   ├── fixtures/              # RAG test fixtures
│   ├── test_finetuning_eval.py
│   ├── test_ocr.py
│   ├── test_pipeline.py       # RAG pipeline offline tests
│   ├── test_providers.py
│   ├── test_security.py
│   └── test_workflow.py       # LangGraph end-to-end tests
├── data/                      # Processed datasets and indexes
├── Dockerfile                 # Production multi-stage Docker container
├── Makefile                   # Unified developer automation with virtualenv
├── MODEL_CARD.md              # Model card & Responsible AI documentation
├── pyproject.toml             # Project metadata
├── requirements.txt           # Unified dependency specification
└── run.py                     # Root CLI runner
```

---

## Quickstart & Verification

### 1. Run Complete Automated Test Suite (Offline, Zero-Cost)
```bash
make test
```
Runs all 36+ automated tests across OCR, security guardrails, multi-provider abstraction, fine-tuning evaluators, LangGraph workflows, and RAG retrieval pipelines.

### 2. Launch FinKYC Verification Web UI & Admin Console
```bash
make serve
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser:
- **User Portal**: Run end-to-end verification across dataset records.
- **Admin Console**: Hot-swap inference providers, adjust similarity thresholds, toggle security guardrails, and inspect benchmark matrices.

### 3. Launch RBI KYC RAG Application
```bash
make serve-rag
```
Or directly:
```bash
streamlit run app/streamlit_app.py
```

### 4. RAG CLI Commands
```bash
# Ingest and parse RBI Directions
python -m kycrag.cli build

# Ask a regulatory compliance question
python -m kycrag.cli ask "What are the limits on a small account?"

# Run RAG ablation evaluation (recall across 5 retrieval configs)
python eval/run_eval.py --retrieval-only
```

---

## Slurm GPU Cluster Training (VM03) & Deployment

For complete instructions, consult [documentation/SERVER_DEPLOYMENT_GUIDE.md](../documentation/SERVER_DEPLOYMENT_GUIDE.md).

### 1. Package & Submit to Cluster
```bash
# Package scripts and KYC datasets locally
make package-slurm

# Copy package to cluster
scp -i <ssh-key> finkyc_slurm_package.tar.gz learner03@<vm03-ip>:~/project/

# Connect and submit Slurm job on VM03
ssh -i <ssh-key> learner03@<vm03-ip>
cd ~/project && tar -xzvf finkyc_slurm_package.tar.gz && mkdir -p logs
sbatch train.sbatch
```

### 2. Verify on Cluster & Deploy Container
```bash
# Verify inference on cluster without Docker
python3 infer_slurm.py --adapter_path ~/checkpoints/job-<JOB_ID>

# Download adapter weights locally & run production Docker container
make docker-build
make docker-run
```

---

## Authorship & Attribution
- **Lead Engineer**: Chandan Bauri (<chandanbauri441@gmail.com>)
- **Repository**: [Banking-KYC](https://github.com/TheSeven-IITKGP/Banking-KYC.git)
- **Active Integration Branch**: `dev`
- **Source Feature Branches (Preserved)**:
  - `feature/kyc-finetuning-pipeline` (Model Adaptation, OCR, Security, LangGraph)
  - `add-kyc-rag-app` (RBI Regulatory RAG Application)
