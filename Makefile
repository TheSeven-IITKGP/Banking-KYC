# FinKYC Platform Makefile
# Lead Engineer: Chandan Bauri <chandanbauri441@gmail.com>

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
PORT ?= 8080
CONFIG ?= config_1_balanced

.PHONY: all help venv install test prepare-data evaluate serve providers train docker-build docker-run clean

# Default target
all: install test prepare-data

# Help banner
help:
	@echo "=================================================================="
	@echo " FinKYC Platform - Make Automation & Virtual Environment Manager"
	@echo " Lead Engineer: Chandan Bauri <chandanbauri441@gmail.com>"
	@echo "=================================================================="
	@echo "Available commands:"
	@echo "  make venv          - Create virtualenv (.venv) if not already present"
	@echo "  make install       - Set up .venv and install/verify dependencies"
	@echo "  make test          - Run full automated test suite inside .venv"
	@echo "  make prepare-data  - Generate train/val/test splits inside .venv"
	@echo "  make evaluate      - Run benchmark evaluation matrix inside .venv"
	@echo "  make serve         - Start API & Web UI on http://localhost:$(PORT)"
	@echo "  make providers     - List configured inference providers inside .venv"
	@echo "  make train         - Run fine-tuning training job (dry run) inside .venv"
	@echo "  make docker-build  - Build production Docker container image"
	@echo "  make docker-run    - Run production container on port $(PORT)"
	@echo "  make clean         - Remove .venv, caches, and temporary build files"
	@echo "=================================================================="

# Rule to auto-create virtual environment if not present
$(VENV)/bin/python3:
	@echo "🔧 Virtual environment not found. Creating $(VENV)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --system-site-packages $(VENV); \
	else \
		python3 -m venv --system-site-packages $(VENV); \
	fi
	@echo "✅ Virtual environment created at $(VENV)."

venv: $(VENV)/bin/python3

# Install dependencies into virtualenv
install: $(VENV)/bin/python3
	@echo "📦 Checking and installing dependencies in $(VENV)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -r requirements.txt || true; \
	else \
		$(PIP) install -r requirements.txt || true; \
	fi
	@$(PYTHON) -c "import pydantic, torch, pandas; print('✅ Environment verified: Core packages loaded successfully in .venv.')"

# Run tests
test: $(VENV)/bin/python3
	@echo "🧪 Running automated test suite inside $(VENV)..."
	@$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

# Prepare instruction tuning datasets
prepare-data: $(VENV)/bin/python3
	@echo "📊 Generating dataset splits and instruction formats inside $(VENV)..."
	@$(PYTHON) run.py prepare-data --output-dir data/processed

# Run benchmark evaluation
evaluate: $(VENV)/bin/python3
	@echo "📈 Running benchmark evaluation matrix inside $(VENV)..."
	@$(PYTHON) run.py evaluate

# Launch web server and API
serve: $(VENV)/bin/python3
	@echo "🚀 Launching FinKYC Web UI and REST API on port $(PORT)..."
	@echo "👉 Open your browser at http://localhost:$(PORT)"
	@$(PYTHON) run.py serve --port $(PORT)

# List providers
providers: $(VENV)/bin/python3
	@$(PYTHON) run.py list-providers

# Train model dry run
train: $(VENV)/bin/python3
	@echo "⚙️ Running QLoRA fine-tuning runner for config $(CONFIG)..."
	@$(PYTHON) run.py train --config $(CONFIG)

# Docker commands
docker-build:
	@echo "🐳 Building Docker container image finkyc-platform:1.0.0..."
	@docker build -t finkyc-platform:1.0.0 .

docker-run:
	@echo "🐳 Running Docker container on port $(PORT)..."
	@docker run -p $(PORT):8080 --name finkyc-service finkyc-platform:1.0.0

# Cleanup
clean:
	@echo "🧹 Cleaning up temporary files and virtual environment..."
	@rm -rf $(VENV) build dist *.egg-info .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Clean complete."
