#!/bin/bash
# Create the quantem tutorial env (separate from the SGFM env at the repo root).
set -e
cd "$(dirname "$0")"
uv sync --python 3.11
.venv/bin/python -m ipykernel install --user --name quantem-tutorials --display-name "Python (quantem-tutorials)"
echo "Done. Activate with: source $(pwd)/.venv/bin/activate"
