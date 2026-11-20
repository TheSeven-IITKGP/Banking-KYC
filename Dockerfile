# Multi-Stage Production Dockerfile for Banking KYC Platform
# Author: chandan bauri <chandanbauri441@gmail.com>

FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run data preparation on build
RUN python run.py prepare-data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/config || exit 1

ENTRYPOINT ["python", "run.py", "serve", "--port", "8080"]
