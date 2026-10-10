#!/bin/sh
# Waits for run_all_joint.sh, then re-runs reduced_s2/B seed 0 (deterministic) so that the S-beam files used by the
# S-response figure exist for that group (the first queue job started before beam saving was added).
cd "$(dirname "$0")"
while [ ! -f results_joint/logs/done.txt ]; do sleep 60; done
python run_joint_probe.py exp --sample reduced_s2 --config B --variants fixed,joint,oracle --seeds 0 > results_joint/logs/reduced_s2_B_seed0_rerun.log 2>&1
echo FOLLOWUP_DONE > results_joint/logs/followup_done.txt
