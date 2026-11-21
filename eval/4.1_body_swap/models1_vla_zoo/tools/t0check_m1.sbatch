#!/bin/bash
#SBATCH --job-name=m1_t0chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# models1 7.2: step-0 consistency (both suites, shard-0 keys) + 7.2 regression: embody_toy (toy switches off) replays the
# published OFT sham_cg / grpkp100 dumps of keys (0,0) and (2,0) with qpos diff 0 (tools/regress_toy_copy.py =
# byte copy of embody_toy/tools/regress.py).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_toy
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/t0check; mkdir -p $O; cd $Z
python tools/t0check_m1.py --suite libero_spatial --out $O/t0_spatial.jsonl > $O/t0_spatial.log 2>&1 &
python tools/t0check_m1.py --suite libero_object --out $O/t0_object.jsonl > $O/t0_object.log 2>&1 &
for t in $( [ "${RUN_REGR:-0}" = 1 ] && echo 0 2 ); do
  env COPY_GRIP_CMD=1 python tools/regress_toy_copy.py --render 1 --code $EMBODY_DIR --cond sham_cg --body Panda --tasks $t --eps 0-0 --out $O/regr_sham_cg_t$t > $O/regr_sham_cg_t$t.log 2>&1 &
  env COPY_GRIP_CMD=1 GRP_KP=100 python tools/regress_toy_copy.py --render 1 --code $EMBODY_DIR --cond grpkp100 --body PandaR85 --tasks $t --eps 0-0 --out $O/regr_grpkp100_t$t > $O/regr_grpkp100_t$t.log 2>&1 &
done
wait; echo done $(date)
