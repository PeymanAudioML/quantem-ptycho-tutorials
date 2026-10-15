#!/usr/bin/env bash
# Joint S-matrix + probe fits, one at a time (no GPU sharing): Au diagnostic, then the three-grain Si specimen.
# Checkpoint every 5 epochs; early stop when the best loss improves < 1 % over 6 epochs (after the 8 warm-up epochs).
set -uo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
COMMON="--heldout 0 --epochs 80 --checkpoint-every 5 --patience 6 --min-rel-improvement 0.01 --verbose --device cuda"
PYTHONUNBUFFERED=1 $PY run_benchmark.py smatrix --specimen A_z100 --tag full80 $COMMON > smatrix_A_z100_full80.log 2>&1
PYTHONUNBUFFERED=1 $PY run_benchmark.py smatrix --config configs/poly_si_3grain.yaml --specimen P3_XY --tag full80 $COMMON > smatrix_P3_XY_full80.log 2>&1
echo CHAIN_DONE >> smatrix_P3_XY_full80.log
