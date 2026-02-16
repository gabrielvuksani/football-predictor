FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (layer cached if pyproject.toml unchanged)
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || true

# Copy source
COPY src/ src/
COPY web/ web/
COPY data/ data/

# Reinstall with source present
RUN pip install --no-cache-dir -e .

# Non-root user for security
RUN useradd -m footy && chown -R footy:footy /app
USER footy

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/last-updated')" || exit 1

CMD ["uvicorn", "web.api:app", "--host", "0.0.0.0", "--port", "8000"]
