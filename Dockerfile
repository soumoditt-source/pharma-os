# PharmaOS — Drug Discovery Molecular Optimization RL Environment
# Dockerfile (project ROOT — required by OpenEnv spec)
#
# Built by: Soumoditya Das (soumoditt@gmail.com)
# Meta x PyTorch OpenEnv Hackathon 2026
#
# Notes:
#   - Uses python:3.11-slim for small image size
#   - RDKit installed from conda-forge mirror (rdkit PyPI wheel)
#   - Multi-stage build not needed; rdkit wheel is self-contained
#   - HF Spaces default port: 7860

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System libraries required by RDKit
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies (rdkit from PyPI wheel)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy project files
# ---------------------------------------------------------------------------
COPY models.py .
COPY client.py .
COPY inference.py .
COPY openenv.yaml .
COPY data/ ./data/
COPY server/ ./server/

# ---------------------------------------------------------------------------
# Ensure server is importable as a package
# ---------------------------------------------------------------------------
RUN touch server/__init__.py

# ---------------------------------------------------------------------------
# Create non-root user for security
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 pharmao
RUN chown -R pharmao:pharmao /app
USER pharmao

# ---------------------------------------------------------------------------
# Environment defaults (overridable at runtime)
# ---------------------------------------------------------------------------
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV PHARMAO_TASK=lipinski_optimizer
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Expose HF Spaces default port
# ---------------------------------------------------------------------------
EXPOSE 7860

# ---------------------------------------------------------------------------
# Health check (validates environment is responsive)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:${PORT}/health',timeout=5); exit(0 if r.status_code==200 else 1)"

# ---------------------------------------------------------------------------
# Launch FastAPI server
# ---------------------------------------------------------------------------
CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --log-level info"]
