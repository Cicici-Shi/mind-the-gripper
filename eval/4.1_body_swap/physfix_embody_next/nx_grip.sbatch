#!/bin/bash
#SBATCH --job-name=nx_grip
#SBATCH --partition=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Gripper-object interaction under replayed SRC actions (no policy): steps with any gripper-object contact and with
# both finger pads on a movable object, for sham Panda, R85 kp100 (G) and R140 kp100 / kp20; spatial + goal, shard 0.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
N=$PROJ_ROOT/embody_next; O=$N/out/grip; mkdir -p $O; cd $N
echo "node=$(hostname) start=$(date) body_env_sha=$(sha256sum embody/body_env.py | cut -c1-16)"
for s in libero_spatial libero_goal; do
  for spec in "sham|Panda|GRP_MAP=kp20 COPY_GRIP_CMD=1" "g|PandaR85|GRP_KP=100 GRP_MAP=kp20 COPY_GRIP_CMD=1" "r140kp100|PandaR140|R140_KP=100 COPY_GRIP_CMD=1" "r140kp20|PandaR140|R140_KP=20 COPY_GRIP_CMD=1"; do
    IFS='|' read t b v <<< "$spec"; mkdir -p $O/$t
    env $v python tools/nx_check.py --code $N --suite $s --body $b --offset 0 --modes replay --out $O/$t > $O/$t/${s}.log 2>&1 &
  done
done
wait
echo "end=$(date)"
