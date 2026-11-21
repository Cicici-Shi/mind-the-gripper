#!/bin/bash
#SBATCH --job-name=v3_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Block-3 rendering checks: twin regression (old dev code vs embody_v3 on 5 dumped episodes of
# sham_twin and grpkp100_twin) and smoke replays of every new condition (SRC actions, closed loop,
# episode 0 of the 10 tasks, screenshots to ~/tmp/v3_smoke).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; D=$PROJ_ROOT/embody_dev; O=$V/out; mkdir -p $O/regress_twin $O/smoke; cd $V
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
run() { local name=$1; shift; env "$@" > $O/$name.log 2>&1 & }
R="python tools/regress.py --render 1 --tasks 0,1,2,3,4 --eps 0-0"
run regress_twin/shamtw_dev TWIN_RENDER=1 $R --code $D --cond sham_twin --body Panda --out $O/regress_twin/shamtw_dev
run regress_twin/shamtw_v3  TWIN_RENDER=1 $R --code $V --cond sham_twin --body Panda --out $O/regress_twin/shamtw_v3
run regress_twin/grptw_dev  TWIN_RENDER=1 GRP_KP=100 $R --code $D --cond grpkp100_twin --body PandaR85 --out $O/regress_twin/grptw_dev
run regress_twin/grptw_v3   TWIN_RENDER=1 GRP_KP=100 $R --code $V --cond grpkp100_twin --body PandaR85 --out $O/regress_twin/grptw_v3
S="python tools/smoke.py --render 1 --out $O/smoke"
run smoke/S     $S --name S --body Panda
run smoke/G     GRP_KP=100 $S --name G --body PandaR85
run smoke/P6j3  $S --name P6j3 --body PandaL3
run smoke/K3    $S --name K3 --body Kinova3
run smoke/RT    $S --name RT --body PandaRT
run smoke/G_R   TWIN_RENDER=1 TWIN_BODY=PandaR85 $S --name G_R --body Panda
run smoke/G_Tw  TWIN_RENDER=1 TWIN_CAMS=wrist GRP_KP=100 $S --name G_Tw --body PandaR85
run smoke/G_Ta  TWIN_RENDER=1 TWIN_CAMS=agent GRP_KP=100 $S --name G_Ta --body PandaR85
run smoke/RT_T  TWIN_RENDER=1 $S --name RT_T --body PandaRT
wait
echo done $(date)
