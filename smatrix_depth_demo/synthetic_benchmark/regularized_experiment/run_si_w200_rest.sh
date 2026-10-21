#!/usr/bin/env bash
# Continuation after the chain was edited (2026-10-20 ~15:12 UTC): the F runs (fixed + trained probe) run in PID $1;
# then QuantEM arm E with the trained probe only (E fixed dropped), plots, then the def2-only joint S-matrix fits.
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
CFG=results/defocus_study/si_defocus_series_w200_window_noiseless/configs/k4_d0-d1-d2-d3.yaml
SP=k4_d0-d1-d2-d3
RUNS=runs/defocus_study_si_defocus_series_w200_window_noiseless/$SP
run() { echo "=== $(date -Is) $*"; "$@"; echo "=== $(date -Is) exit $?"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== $(date -Is) F runs finished"
run $PY run_benchmark.py quantem --config $CFG --specimen $SP --dataset 2 --smatrix-run smatrix_window_noiseless_s0_study \
  --probes smatrix_trained --iters 200 --probe-warmup 40 --qlr 5e-3 --probe-lr 1e-4 --qbatch 64 --seeds 0 --device cuda \
  --arms E --tag k4QE
for r in $RUNS/quantem_*_k4Q32 $RUNS/quantem_*_k4QE; do
  [ -f $r/summary.json ] && run $PY plot_quantem.py --config $CFG --specimen $SP --run $(basename $r)
done
echo "=== $(date -Is) QuantEM done; starting def2-only joint S-matrix fits"
bash regularized_experiment/run_si_w200_d2_joint.sh > regularized_experiment/si_w200_d2_joint.log 2>&1
echo "=== $(date -Is) all done"
