#!/bin/bash
#SBATCH --job-name=m1_fscheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1 EMBODY_DIR=$PROJ_ROOT/embody_toy; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $PROJ_ROOT/vla_zoo
for spec in "sham_twin 0" "g_tfc 0" "g_tw_fc 2" "g_ta_fc 2"; do set -- $spec
  env $(tools/zoo_cells.sh vars $1 | cut -d" " -f2-) BODY=$(tools/zoo_cells.sh vars $1 | cut -d" " -f1) python tools/fastsync_check.py --task $2 --steps 80 > out/fscheck_$1.log 2>&1 &
done; wait; grep -h FASTSYNC_CHECK out/fscheck_*.log
