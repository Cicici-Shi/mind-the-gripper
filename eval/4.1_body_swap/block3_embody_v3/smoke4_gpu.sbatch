#!/bin/bash
#SBATCH --job-name=v3_smoke4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# G_R sunk v2 (pad target = actual physical pad centre incl. lateral offset); rendering only, no policy.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 COPY_GRIP_CMD=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/smoke4; mkdir -p $O; cd $V
for e in 0 1 2; do n=0; [ $e = 0 ] && n=5
  env TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk python tools/smoke.py --render 1 --out $O --shots $HOME/tmp/v3_smoke/b3v2 --name G_R_sunk --body Panda --ep $e --nshots $n --save-t0 $O/t0_G_R > $O/G_R_e$e.log 2>&1 &
done; wait; echo done
