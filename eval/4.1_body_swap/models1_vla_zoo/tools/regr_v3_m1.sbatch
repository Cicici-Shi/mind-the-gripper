#!/bin/bash
#SBATCH --job-name=m1_regr_v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.log
# models1 7.2 regression, old side: same replay as t0check_m1.sbatch but with code = embody_v3 (read-only import),
# so the embody_toy replay is compared with the embody_v3 replay step by step (toy1 3.1/3.4 method).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/t0check; V=$PROJ_ROOT/embody_v3; cd $Z
for t in 0 2; do
  env EMBODY_DIR=$V COPY_GRIP_CMD=1 python tools/regress_toy_copy.py --render 1 --code $V --cond sham_cg --body Panda --tasks $t --eps 0-0 --out $O/regrv3_sham_cg_t$t > $O/regrv3_sham_cg_t$t.log 2>&1 &
  env EMBODY_DIR=$V COPY_GRIP_CMD=1 GRP_KP=100 python tools/regress_toy_copy.py --render 1 --code $V --cond grpkp100 --body PandaR85 --tasks $t --eps 0-0 --out $O/regrv3_grpkp100_t$t > $O/regrv3_grpkp100_t$t.log 2>&1 &
done
wait; echo done $(date)
