#!/bin/bash
#SBATCH --job-name=nx_smoke_cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# embody_next smoke (no policy): a few keys per scene type through tools/nx_check.py, physics only.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
N=$PROJ_ROOT/embody_next; O=$N/out/smoke_$SLURM_JOB_ID; mkdir -p $O; cd $N
echo "node=$(hostname) start=$(date) body_env_sha=$(sha256sum embody/body_env.py | cut -c1-16)"
C="python tools/nx_check.py --code $N --n 1 --offset 0 --zero-steps 60 --out $O"
G="GRP_KP=100 GRP_MAP=kp20 COPY_GRIP_CMD=1"
env COPY_GRIP_CMD=1 $C --suite libero_10 --body SRC --tasks 0,4,9 > $O/src.log 2>&1 &
env $G $C --suite libero_10 --body PandaR85 --tasks 0,4,9 > $O/g10.log 2>&1 &
env COPY_GRIP_CMD=1 $C --suite libero_10 --body Kinova3 --tasks 0,4,9 > $O/k3_10.log 2>&1 &
env COPY_GRIP_CMD=1 $C --suite libero_10 --body UR5e --tasks 0,4,9 > $O/ur5e_10.log 2>&1 &
env $G $C --suite libero_goal --body PandaR85 --tasks 0,1 > $O/ggoal.log 2>&1 &
env COPY_GRIP_CMD=1 $C --suite libero_goal --body PandaL3 --tasks 0 > $O/l3goal.log 2>&1 &
env COPY_GRIP_CMD=1 $C --suite libero_goal --body PandaRT2 --tasks 0 > $O/rt2goal.log 2>&1 &
wait
echo "end=$(date)"
