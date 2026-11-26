
SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PORT ?= 8080
CONFIG ?= config_1_balanced

.PHONY: all help venv install test prepare-data evaluate serve serve-rag serve-ft providers train docker-build docker-run package-slurm clean

all: install test prepare-data

help:
	@echo "=================================================================="
	@echo " FinKYC Platform - Make Automation & Virtual Environment Manager"
	@echo " Lead Engineer: Chandan Bauri <chandanbauri441@gmail.com>"
	@echo "=================================================================="
	@echo "Available commands:"
	@echo "  make venv          - Create virtualenv (.venv) if not already present"
	@echo "  make install       - Set up .venv and install/verify dependencies"
	@echo "  make test          - Run full unified automated test suite (36 tests) inside .venv"
	@echo "  make prepare-data  - Generate train/val/test splits inside .venv"
	@echo "  make evaluate      - Run fine-tuning benchmark evaluation matrix inside .venv"
	@echo "  make serve         - Start FinKYC API & Web UI on http://localhost:$(PORT)"
	@echo "  make serve-rag     - Start RBI KYC Regulatory RAG Streamlit App on http://localhost:8501"
	@echo "  make serve-ft      - Start KYC Fine-Tuning Verification Streamlit App on http://localhost:8502"
	@echo "  make build-rag     - Parse and build RAG corpus vector & BM25 index"
	@echo "  make eval-rag      - Run RAG ablation benchmark (5 retrieval configs)"
	@echo "  make providers     - List configured inference providers inside .venv"
	@echo "  make train         - Run fine-tuning training job (dry run) inside .venv"
	@echo "  make package-slurm - Bundle training files into finkyc_slurm_package.tar.gz for VM03"
	@echo "  make docker-build  - Build production Docker container image"
	@echo "  make docker-run    - Run production container on port $(PORT)"
	@echo "  make clean         - Remove .venv, caches, and temporary build files"
	@echo "=================================================================="

$(VENV)/bin/python3:
	@echo "🔧 Virtual environment not found. Creating $(VENV)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --system-site-packages $(VENV); \
	else \
		python3 -m venv --system-site-packages $(VENV); \
	fi
	@echo "✅ Virtual environment created at $(VENV)."

venv: $(VENV)/bin/python3

install: $(VENV)/bin/python3
	@echo "📦 Checking and installing dependencies in $(VENV)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -r requirements.txt || true; \
	else \
		$(PIP) install -r requirements.txt || true; \
	fi
	@$(PYTHON) -c "import pydantic, torch, pandas; print('✅ Environment verified: Core packages loaded successfully in .venv.')"

test: $(VENV)/bin/python3
	@echo "🧪 Running full automated test suite (36 tests) inside $(VENV)..."
	@PYTHONPATH=.:src $(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

serve-rag: $(VENV)/bin/python3
	@echo "📚 Launching RBI KYC Directions RAG App on http://127.0.0.1:8501..."
	@PYTHONPATH=.:src $(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1

serve-ft: $(VENV)/bin/python3
	@echo "🔍 Launching KYC Fine-Tuning Verification UI on http://127.0.0.1:8502..."
	@PYTHONPATH=.:src $(PYTHON) -m streamlit run app/finetuning_streamlit_app.py --server.port 8502 --server.address 127.0.0.1

download-rag: $(VENV)/bin/python3
	@echo "📥 Fetching documents from manifest.csv into data/raw/..."
	@PYTHONPATH=.:src $(PYTHON) -m kycrag.cli download

build-rag: $(VENV)/bin/python3
	@if [ ! -d "data/raw" ] || [ -z "$$(ls -A data/raw 2>/dev/null)" ]; then \
		echo "📥 data/raw is empty. Fetching source documents from manifest.csv..."; \
		PYTHONPATH=.:src $(PYTHON) -m kycrag.cli download || \
		(echo "⚠️ Automated download blocked. If needed, save the PDFs manually to data/raw/ or use the pre-built index in data/processed/." && exit 1); \
	fi
	@echo "🏗️ Building RAG corpus index..."
	@PYTHONPATH=.:src $(PYTHON) -m kycrag.cli build

eval-rag: $(VENV)/bin/python3
	@echo "📊 Running RAG ablation benchmark across 5 retrieval configs..."
	@PYTHONPATH=.:src $(PYTHON) eval/run_eval.py --retrieval-only

prepare-data: $(VENV)/bin/python3
	@echo "📊 Generating dataset splits and instruction formats inside $(VENV)..."
	@$(PYTHON) run.py prepare-data --output-dir data/processed

evaluate: $(VENV)/bin/python3
	@echo "📈 Running benchmark evaluation matrix inside $(VENV)..."
	@$(PYTHON) run.py evaluate

serve: $(VENV)/bin/python3
	@echo "🚀 Launching FinKYC Web UI and REST API on port $(PORT)..."
	@echo "👉 Open your browser at http://localhost:$(PORT)"
	@$(PYTHON) run.py serve --port $(PORT)

providers: $(VENV)/bin/python3
	@$(PYTHON) run.py list-providers

train: $(VENV)/bin/python3
	@echo "⚙️ Running QLoRA fine-tuning runner for config $(CONFIG)..."
	@$(PYTHON) run.py train --config $(CONFIG)

package-slurm:
	@echo "📦 Packaging FinKYC Slurm training artifacts for VM03 cluster..."
	@tar -czvf finkyc_slurm_package.tar.gz \
		train.sbatch \
		train_slurm.py \
		infer.sbatch \
		infer_slurm.py \
		banking_kyc \
		data/processed/kyc_train.jsonl \
		data/processed/kyc_val.jsonl \
		data/processed/kyc_test.jsonl
	@echo "✅ Packaged successfully: finkyc_slurm_package.tar.gz"
	@echo "👉 Copy to cluster with: scp -i <ssh-key> finkyc_slurm_package.tar.gz learner03@<vm03-ip>:~/project/"

docker-build:
	@echo "🐳 Building Docker container image finkyc-platform:1.0.0..."
	@docker build -t finkyc-platform:1.0.0 . 2>/dev/null || \
	 (echo "⚠️ Docker buildx permission issue detected. Retrying with legacy builder (DOCKER_BUILDKIT=0)..." && \
	  DOCKER_BUILDKIT=0 docker build -t finkyc-platform:1.0.0 . 2>/dev/null) || \
	 (echo "❌ Docker build failed. If you see 'permission denied' on ~/.docker/buildx, run:" && \
	  echo "   sudo chown -R \$$(whoami) ~/.docker" && exit 1)

docker-run:
	@echo "🐳 Running Docker container on port $(PORT)..."
	@docker run -p $(PORT):8080 --name finkyc-service finkyc-platform:1.0.0

clean:
	@echo "🧹 Cleaning up temporary files and virtual environment..."
	@rm -rf $(VENV) build dist *.egg-info .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Clean complete."
