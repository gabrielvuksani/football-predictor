FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || true

# Copy source
COPY src/ src/
COPY web/ web/
COPY data/ data/

# Reinstall with source present
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "web.api:app", "--host", "0.0.0.0", "--port", "8000"]
