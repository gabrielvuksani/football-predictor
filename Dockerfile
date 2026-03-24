# Football Predictor v14 — "Apex"
# Multi-stage build for minimal image size
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System deps for scraping
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY web/ web/

# Install project in editable mode
RUN pip install --no-cache-dir .

# Ensure data directory exists
RUN mkdir -p data data/models

# Non-root user
RUN useradd -m -s /bin/bash footy && chown -R footy:footy /app
USER footy

EXPOSE 8000

# Health check on the proper endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "web.api:app", "--host", "0.0.0.0", "--port", "8000"]
