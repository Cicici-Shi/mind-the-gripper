#!/bin/bash
#SBATCH --job-name=pf_dscan
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# physfix rev3 7.1 / 5.5: full detector-D rescan of the 12 pf_ cells (numpy only, never reads success).
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo
echo "node=$(hostname) start=$(date) scan=$(sha256sum tools/physics_breakdown_scan.py|cut -c1-16)"
python tools/physics_breakdown_scan.py --preset physfix --out out/physics_breakdown_physfix.json
echo "end=$(date) rc=$?"
