#!/usr/bin/env bash
# Footy Predictor — Quick Setup
# Usage:  bash bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== Footy Predictor Setup ==="

# 1. Create data directories
mkdir -p data/models logs

# 2. Create .env from example if it doesn't exist
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "[✓] Created .env from .env.example — please edit it with your API keys"
    else
        echo "[!] No .env.example found. Create a .env file with your API keys."
    fi
else
    echo "[✓] .env already exists"
fi

# 3. Create virtual environment
if [[ ! -d .venv ]]; then
    echo "[…] Creating Python virtual environment…"
    python3 -m venv .venv
    echo "[✓] Virtual environment created"
else
    echo "[✓] Virtual environment already exists"
fi

# 4. Activate and install
echo "[…] Installing package…"
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys:"
echo "       FOOTBALL_DATA_ORG_TOKEN=your_token_here"
echo "       API_FOOTBALL_KEY=your_key_here"
echo ""
echo "  2. Activate the environment:"
echo "       source .venv/bin/activate"
echo ""
echo "  3. Run the full pipeline:"
echo "       footy go"
echo ""
echo "  4. Launch the web UI:"
echo "       footy serve"
