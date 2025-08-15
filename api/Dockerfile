# Lightweight Dockerfile for the Rho FastAPI backend (rho/api)
# - Uses python:3.11-slim as a small base
# - Installs requirements from requirements.txt
# - Copies app code and runs uvicorn
# - Keep image reasonably small and secure for a demo
FROM python:3.11-slim

# Keep Python outputs unbuffered (helpful for logs)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install OS-level deps required for building some Python wheels (kept minimal)
# and for possible future DB drivers (libpq-dev).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Use a non-root user for better security
RUN groupadd --gid 1000 appuser || true && \
    useradd --uid 1000 --gid 1000 --shell /bin/false --no-create-home appuser || true && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Simple healthcheck (optional). If you prefer Docker HEALTHCHECK, uncomment below.
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD curl -f http://127.0.0.1:8000/healthz || exit 1

# Default command: run uvicorn on startup
# Note: in production you may want to use a process manager or configure workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
