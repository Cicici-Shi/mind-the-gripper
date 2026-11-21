#!/bin/bash
#SBATCH --job-name=pf_readout
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# physfix read-out (protocol_physfix.md rev6 c8bd3fd8 §8): ONE run after prereg + protocol freeze (§9.3 rev6, not later than 11/10 18:00 SGT).
# Gate json = rebuilt t0check <job> (WORKLOG_physfix 6.1); gate0 = <job> (<job> void, sealed); grasp = rebuilt <job> json (v1_withcounts sealed).
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
Z=$PROJ_ROOT/vla_zoo; A=$Z/analysis; T=$Z/out/pf/t0check_<job>
echo "readout start $(date) host $(hostname)"
sha256sum $A/protocol_physfix.md $A/prereg_physfix.py $A/prereg_models1.py $A/test_prereg_physfix.py $A/run_prereg_physfix.sbatch
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $A
python prereg_physfix.py --out $A/results_physfix.txt --verdict-json $A/verdict_physfix.json \
  --protocol $A/protocol_physfix.md \
  --gate-json $T/gate_physfix_t0.json --gate-jsonl $T/t0_spatial.jsonl $T/t0_object.jsonl \
  --gate0-json $Z/out/pf/gate0/gate0_physfix_<job>.json \
  --breakdown-json $Z/out/physics_breakdown_physfix.json \
  --padlr-json $Z/out/pf/kp20_padlr/kp20_pad_lr.json \
  --grasp-json $Z/out/pf/grasp_<job>/grasp_physfix.json \
  --geo-summary $Z/out/pf/geo_<job>/summary.json \
  --bg-diag $A/t0check_bg_diag_2026-11-07.md \
  --r85-xml $HOME/miniforge3/envs/oft/lib/python3.10/site-packages/robosuite/models/assets/grippers/robotiq_gripper_85.xml \
  --map-kp20 $PROJ_ROOT/embody_next/embody/body_env.py
echo "exit $? end $(date)"
