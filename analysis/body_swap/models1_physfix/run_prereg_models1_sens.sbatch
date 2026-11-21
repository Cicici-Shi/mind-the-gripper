#!/bin/bash
#SBATCH --job-name=m1_readout_sens
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
# models1 physics-breakdown sensitivity read-out (Fable add1 §2.3 item 2, §4 item 2). ONE run, submitted with
#   sbatch --dependency=afterok:<job> --begin=2026-11-09T12:00:00 run_prereg_models1_sens.sbatch
# i.e. strictly after the frozen main read-out (run_prereg_models1.sbatch, job <job>) succeeded.
# 1. physics-breakdown rescan (tools/physics_breakdown_scan.py, preset m1: all OFTi shards, all pi0 / OpenVLA client logs;
#    never reads success); 2. sha256 of both analysis scripts, the diff and both json; 3. the sens copy with the same
#    remaining arguments as the frozen read-out. Outputs results_models1_sens_physics.txt / verdict_models1_sens_physics.json.
set -u
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$HOME/tmp
Z=$PROJ_ROOT/vla_zoo; A=$Z/analysis
EX=$Z/out/physics_breakdown_m1.json; BD=$Z/out/physics_breakdown_m1_logs.json
echo "sens readout start $(date) host $(hostname)"
cd $Z || exit 1
for f in $EX $BD; do [ -f $f ] && cp -p $f ${f%.json}.prev_$(date +%m%d%H%M).json; done
sha256sum tools/physics_breakdown_scan.py
$HOME/miniforge3/bin/python3 tools/physics_breakdown_scan.py --out $EX --preset m1 || { echo "scan (npz) failed rc $?"; exit 1; }
$HOME/miniforge3/bin/python3 tools/physics_breakdown_scan.py --logs --out $BD || { echo "scan (logs) failed rc $?"; exit 1; }
echo "==== sha256"
sha256sum $A/prereg_models1.py $A/prereg_models1_sens.py $A/prereg_models1_sens.diff $EX $BD \
  $A/make_prereg_models1_sens.py $A/test_prereg_models1_sens.py $A/run_prereg_models1_sens.sbatch $Z/out/gate_m1.json
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $A
python prereg_models1_sens.py --out $A/results_models1_sens_physics.txt --verdict-json $A/verdict_models1_sens_physics.json \
  --gate-json $Z/out/gate_m1.json --exclude-keys $EX --bounds $BD
echo "exit $? end $(date)"
