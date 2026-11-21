#!/bin/bash
#SBATCH --job-name=nx_calib140
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
CODE=${1:-$PROJ_ROOT/embody_next}
echo "node=$(hostname) code=$CODE body_env_sha=$(sha256sum $CODE/embody/body_env.py | cut -c1-16)"
python $CODE/tools/nx_calib_r140.py $CODE $PROJ_ROOT/embody_next/out/calib_r140_$SLURM_JOB_ID.json 2>&1 | grep -v -i "gym\|warning\|migration\|upgrade"
