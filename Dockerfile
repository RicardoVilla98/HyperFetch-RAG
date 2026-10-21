# ==============================================================================
# HyperFetch-RAG: Production Multi-Stage Dockerfile
# Base: Python 3.11-slim
# ==============================================================================

FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-privileged service user
RUN groupadd -g 1001 hyperfetch && \
    useradd -u 1001 -g hyperfetch -s /bin/bash -m hyperfetch

WORKDIR /app

# Install Python dependencies first for caching optimization
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application source code and configuration
COPY src/ /app/src/
COPY .env.example /app/.env.example

# Set up local directory permissions
RUN mkdir -p /app/data /app/.cache && \
    chown -R hyperfetch:hyperfetch /app

# Switch to non-root execution
USER hyperfetch

# Expose FastAPI service port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default runtime entrypoint
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
