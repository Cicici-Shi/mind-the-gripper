#!/bin/bash
#SBATCH --job-name=v3_smoke3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Block-3 rendering smoke after the frozen protocol: G_R sunk twin (30 episodes, 5 with grasp screenshots,
# step-0 images vs G), RT2 and UR5e_nc (task 3, screenshots), all COPY_GRIP_CMD=1.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 COPY_GRIP_CMD=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/smoke3; mkdir -p $O; cd $V
SH=$HOME/tmp/v3_smoke/b3
S="python tools/smoke.py --render 1 --out $O --shots $SH"
for e in 0 1 2; do
  n=0; [ $e = 0 ] && n=5
  env TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk $S --name G_R_sunk --body Panda --ep $e --nshots $n --save-t0 $O/t0_G_R > $O/G_R_e$e.log 2>&1 &
done
env GRP_KP=100 $S --name G --body PandaR85 --ep 0 --nshots 0 --save-t0 $O/t0_G > $O/G.log 2>&1 &
env $S --name RT2 --body PandaRT2 --ep 0 --nshots 2 > $O/RT2.log 2>&1 &
env NO_FOREARM_CONTACT=1 $S --name UR5e_nc --body UR5e --tasks 3 --ep 0,1,2,3,4 --nshots 2 > $O/UR5e_nc.log 2>&1 &
env $S --name UR5e_cg --body UR5e --tasks 3 --ep 0,1,2,3,4 --nshots 1 > $O/UR5e_cg.log 2>&1 &
wait; echo done $(date)
