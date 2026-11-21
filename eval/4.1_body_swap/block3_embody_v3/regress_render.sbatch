#!/bin/bash
#SBATCH --job-name=v3_regr_r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# Same as regress_cpu but with rendering envs (as in the eval): does render mode change the replay?
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/regress_render; mkdir -p $O; cd $V
R="python tools/regress.py --render 1 --eps 0-4"
env $R --code $V --cond src --body SRC --out $O/src_v3 > $O/src_v3.log 2>&1 &
env $R --code $PROJ_ROOT/embody --cond sham --body Panda --out $O/sham_prod > $O/sham_prod.log 2>&1 &
env $R --code $V --cond sham --body Panda --out $O/sham_v3 > $O/sham_v3.log 2>&1 &
wait; echo done $(date)
