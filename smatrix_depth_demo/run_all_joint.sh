#!/bin/sh
# Full experiment queue for the joint S-matrix / probe study (sequential; ~5 h on 4 CPU cores). Logs in results_joint/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p results_joint/logs
R="python run_joint_probe.py exp"
$R --sample reduced_s2 --config B --variants fixed,joint,oracle --seeds 0,1,2,3,4 > results_joint/logs/reduced_s2_B.log 2>&1
$R --sample reduced_s4 --config B --variants fixed,joint,oracle --seeds 0,1,2,3,4 > results_joint/logs/reduced_s4_B.log 2>&1
$R --sample reduced_s2 --config A --variants fixed,joint --seeds 0,1,2 > results_joint/logs/reduced_s2_A.log 2>&1
$R --sample reduced_s2 --config G --variants fixed,joint,oracle --seeds 0 > results_joint/logs/reduced_s2_G.log 2>&1
$R --sample layers3 --config A --variants fixed,joint --seeds 0 > results_joint/logs/layers3_A.log 2>&1
$R --sample layers3 --config B --variants fixed,joint,oracle --seeds 0 > results_joint/logs/layers3_B.log 2>&1
$R --sample six_phi1p5 --config B --variants fixed,joint,oracle --seeds 0 > results_joint/logs/six_B.log 2>&1
$R --sample poly --config B --variants fixed,joint,oracle --seeds 0 > results_joint/logs/poly_B.log 2>&1
$R --sample reduced_s1 --config B --variants fixed,joint,oracle --seeds 0 > results_joint/logs/reduced_s1_B.log 2>&1
$R --sample reduced_s2 --config B --variants oracle,joint --seeds 0,1 --tag _bf --sched '{"det_mask": "bf"}' > results_joint/logs/reduced_s2_B_bf.log 2>&1
echo ALL_DONE > results_joint/logs/done.txt
