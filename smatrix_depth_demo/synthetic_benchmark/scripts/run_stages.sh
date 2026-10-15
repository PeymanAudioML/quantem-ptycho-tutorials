#!/usr/bin/env bash
# Benchmark stages 1-3 (configs/benchmark_200A_K96.yaml) on a GPU machine.  Run from smatrix_depth_demo/synthetic_benchmark.
#
#   ABTEM_PY=python QUANTEM_PY=../../.venv/bin/python bash scripts/run_stages.sh time      # 1 S-matrix fit: measure cost
#   ABTEM_PY=python QUANTEM_PY=../../.venv/bin/python bash scripts/run_stages.sh stage1    # A_z100: S-matrix + QuantEM
#   ABTEM_PY=python QUANTEM_PY=../../.venv/bin/python bash scripts/run_stages.sh stage3    # A_z40/160, B1-3, C
#
# ABTEM_PY: an environment with abtem + torch (data generation).  QUANTEM_PY: QuantEM's environment (reconstruction,
# evaluation).  Finished data are skipped; reconstructions are rerun (they overwrite their own run directory).
set -euo pipefail
ABTEM_PY=${ABTEM_PY:-python}
QUANTEM_PY=${QUANTEM_PY:-python}
DEV=${DEV:-cuda}
ALLPROBES=true_fixed,nominal_fixed,smatrix_fixed,nominal_trained,smatrix_trained

gen()  { $ABTEM_PY generate_data.py --specimens "$1"; }
smx()  { $QUANTEM_PY run_benchmark.py smatrix --specimen "$@" --device $DEV; }
qem()  { $QUANTEM_PY run_benchmark.py quantem --specimen "$@" --device $DEV; }
eval_() { $QUANTEM_PY evaluate_reconstruction.py smatrix --specimen "$1"; $QUANTEM_PY evaluate_reconstruction.py quantem --specimen "$1"; }

case "${1:-}" in
  time)
    $ABTEM_PY check_geometry.py
    gen A_z100
    time smx A_z100 --verbose
    ;;
  stage1)
    gen A_z100
    $QUANTEM_PY validate_forward.py --specimen A_z100 --device cpu
    smx A_z100                                                   # noiseless, seed 0
    smx A_z100 --seeds 1 --init-noise 0.1 --tag init01            # sensitivity to initialisation
    smx A_z100 --noise poisson --seeds 0,1,2                      # sensitivity to noise
    qem A_z100                                                    # arms A-D, probe from the S-matrix fit, trained
    qem A_z100 --arms A,D --probes $ALLPROBES                     # probe / object separation
    qem A_z100 --arms A,D --candidates reliable --tag filtered    # confidence-filtered priors
    eval_ A_z100
    ;;
  stage3)
    for s in A_z40 A_z160 B1_50_150 B2_70_130 B3_85_115 C_stacked C_separated; do
      gen $s
      smx $s
      qem $s
      eval_ $s
    done
    ;;
  *)
    echo "usage: $0 time|stage1|stage3"; exit 1;;
esac
