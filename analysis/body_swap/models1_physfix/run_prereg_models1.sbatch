#!/bin/bash
#SBATCH --job-name=m1_readout
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# models1 read-out (protocol_models1.md 8): ONE run, submitted with --begin=2026-11-09T12:00:00 (SGT).
# Prints the sha256 of every frozen file first, then runs the frozen analysis script. Gate json = t0check (7.2).
set -u
export PYTHONDONTWRITEBYTECODE=1
Z=$PROJ_ROOT/vla_zoo; A=$Z/analysis
echo "readout start $(date) host $(hostname)"
sha256sum $A/protocol_models1.md $A/protocol_models1_decisions.md $A/prereg_models1.py $Z/tools/zoo_cells.sh \
  $Z/tools/zoo_gate_m1.py $Z/tools/t0check_m1.py $Z/tools/t0check_m1_gatejson.py $Z/common/m1_fastsync.py \
  $Z/common/zoo_client.py $Z/oft/run_libero_eval_m1.py $Z/oft/launch_m1.py $Z/out/gate_m1.json
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $A
python prereg_models1.py --out $A/results_models1.txt --verdict-json $A/verdict_models1.json --gate-json $Z/out/gate_m1.json
echo "exit $? end $(date)"
