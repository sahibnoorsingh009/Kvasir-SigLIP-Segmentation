#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p /workspace/data/kvasir-seg
mkdir -p splits checkpoints results/{audit,history,predictions,qualitative,statistics}
echo "Environment installed."
