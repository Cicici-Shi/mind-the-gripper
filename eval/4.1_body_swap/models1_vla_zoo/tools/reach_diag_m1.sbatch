#!/bin/bash
#SBATCH -p normal
#SBATCH --job-name=reach_m1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reach_diag_m1_%j.log
# Usage: sbatch reach_diag_m1.sbatch verify|diag [extra args]
export TMPDIR=$HOME/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
export EMBODY_DIR=$PROJ_ROOT/embody_toy
source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
cd $HOME/tmp
sha256sum $PROJ_ROOT/vla_zoo/tools/reach_diag_m1.py $PROJ_ROOT/vla_zoo/tools/reach_diag_m1.sbatch $EMBODY_DIR/embody/body_env.py $EMBODY_DIR/embody/align.py
python -u $PROJ_ROOT/vla_zoo/tools/reach_diag_m1.py "$@" 2>&1 | grep -v "robosuite WARNING"
