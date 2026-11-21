#!/bin/bash
#SBATCH --job-name=v3_smoke_gr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# G_R smoke v2 (IK warm start from previous twin solution + seed library), 30 episodes; plus sham_twin
# 5-episode regression again (the seed-library change must not touch the Panda-twin path).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out; cd $V; mkdir -p $O/smoke_gr2 $O/regress_twin2
for e in 0 1 2; do env TWIN_RENDER=1 TWIN_BODY=PandaR85 python tools/smoke.py --render 1 --name G_R --body Panda --ep $e --out $O/smoke_gr2 --shots $O/smoke_gr2/shots > $O/smoke_gr2/e$e.log 2>&1 & done
for c in "shamtw $PROJ_ROOT/embody_dev sham_twin Panda" "shamtw3 $V sham_twin Panda"; do set -- $c
  env TWIN_RENDER=1 python tools/regress.py --render 1 --tasks 0,1,2,3,4 --eps 0-0 --code $2 --cond $3 --body $4 --out $O/regress_twin2/$1 > $O/regress_twin2/$1.log 2>&1 & done
wait; echo done
