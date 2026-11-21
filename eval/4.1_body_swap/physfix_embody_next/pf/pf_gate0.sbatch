#!/bin/bash
#SBATCH --job-name=pf_gate0
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# physfix rev3 post-hoc slice gates (blind; CPU; no outcome printed). Output out/pf/gate0/gate0_physfix_<job>.json
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $PROJ_ROOT/vla_zoo
echo "node=$(hostname) start=$(date) script=$(sha256sum next/pf/pf_gate0.py|cut -c1-16)"
python next/pf/pf_gate0.py --out out/pf/gate0/gate0_physfix_$SLURM_JOB_ID.json
echo "end=$(date) rc=$?"
