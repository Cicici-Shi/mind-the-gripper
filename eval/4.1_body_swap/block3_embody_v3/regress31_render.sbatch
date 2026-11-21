#!/bin/bash
#SBATCH --job-name=v3_regr31_r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# 2026-10-31 render-path regression (twin painting) old vs new, and a smoke run of tools/gr_t0check.py.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; B=$V/backup_20261031; O=$V/out/regress31r; mkdir -p $O; cd $V
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
R="python tools/regress.py --render 1"
env TWIN_RENDER=1 $R --code $B --cond sham_twin --body Panda --eps 0-0 --out $O/twin_old > $O/twin_old.log 2>&1 &
env TWIN_RENDER=1 $R --code $V --cond sham_twin --body Panda --eps 0-0 --out $O/twin_new --contacts 1 > $O/twin_new.log 2>&1 &
G="TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk COPY_GRIP_CMD=1"
env $G $R --code $B --runs $V/out/entry2_oft --cond libero_spatial_g_r_sunk --body Panda --tasks 0 --eps 0-0 --out $O/gr_old > $O/gr_old.log 2>&1 &
env $G $R --code $V --runs $V/out/entry2_oft --cond libero_spatial_g_r_sunk --body Panda --tasks 0 --eps 0-0 --out $O/gr_new --contacts 1 > $O/gr_new.log 2>&1 &
wait
python -X faulthandler tools/gr_t0check.py --offsets 0 --tasks 0,2 --trials 2 --out $O/gr_t0check_smoke.jsonl --save-png $O/gr_t0png > $O/gr_t0check_smoke.log 2>&1
echo "gr_t0check exit=$?"
echo done $(date)
