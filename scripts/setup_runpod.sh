#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python scripts/download_checkpoints.py
python scripts/prepare_examples.py
echo 'Setup complete. Start with: python app.py'
