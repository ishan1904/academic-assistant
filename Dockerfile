# ─────────────────────────────────────────────
# Stage 1: Base image
# ─────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# poppler-utils → PDF processing
# libmagic1     → file type detection
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Install Python dependencies
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copy application code
# ─────────────────────────────────────────────
COPY . .

# Create data directory for SQLite fallback
RUN mkdir -p /app/data

# ─────────────────────────────────────────────
# Expose port
# ─────────────────────────────────────────────
EXPOSE 5000

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ─────────────────────────────────────────────
# Start command
# Uses $PORT env var (Railway/AWS/Azure set this automatically)
# Falls back to 5000 for local Docker runs
# ─────────────────────────────────────────────
CMD gunicorn app:app -b 0.0.0.0:${PORT:-8080} --workers 2 --timeout 180