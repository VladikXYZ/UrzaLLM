#!/usr/bin/env bash
# UrzaLLM embedding: sets up a local Python environment and runs embed.py.
# Usage: bash run.sh            (extra arguments go to embed.py, e.g. bash run.sh --dry-run)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (on Linux this includes NVIDIA GPU support; on a Mac it uses Apple GPU)..."
    python -m pip install --upgrade pip
    python -m pip install torch
fi
python -m pip install -r requirements.txt

python embed.py "$@"
