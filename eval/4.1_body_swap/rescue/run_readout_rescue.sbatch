#!/bin/bash
#SBATCH --job-name=rs_readout
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# rescue pilot read-out (protocol_rescue_pilot.md 78c1e1d3, frozen 2026-11-08 14:34): ONE run, all 6 cell dirs complete
# (<job> / <job> COMPLETED 2026-11-09, 200 npz per cell; smoke check <job> PASS). Run by main session 2026-11-12.
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
echo "readout start $(date) host $(hostname)"
cd $PROJ_ROOT/rescue
sha256sum protocol_rescue_pilot.md readout_rescue.py test_readout_rescue.py run_readout_rescue.sbatch $PROJ_ROOT/vla_zoo/analysis/prereg_models1.py
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
python readout_rescue.py --out results_rescue.txt --json verdict_rescue.json
echo "exit $? end $(date)"
