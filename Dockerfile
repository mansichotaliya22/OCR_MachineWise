# ============================================================
# Industrial OCR Monitoring System — Dockerfile
# ============================================================
# Multi-stage build is not used here because all dependencies
# (PaddlePaddle, YOLOv8, OpenCV) are binary wheels that must
# remain in the final image.  We optimise layer caching instead.

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
# libgl1 + libglib2.0-0: required by OpenCV headless
# libgomp1: required by PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer — only re-runs when requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────
RUN mkdir -p images/input results models detector

# ── Expose API port ───────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Default command ───────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
