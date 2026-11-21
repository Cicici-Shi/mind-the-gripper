#!/bin/bash
#SBATCH --job-name=v3_regress
#SBATCH --partition=normal
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Block-3 merge regression, physics only (CPU): replays dumped actions with old and merged code.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/regress; mkdir -p $O; cd $V
P=$PROJ_ROOT/embody; D=$PROJ_ROOT/embody_dev
run() { local name=$1; shift; env "$@" > $O/$name.log 2>&1 & }
R="python tools/regress.py --eps 0-4"
run src_v3        $R --code $V --cond src  --body SRC   --out $O/src_v3
run src_prod      $R --code $P --cond src  --body SRC   --out $O/src_prod
run sham_prod     $R --code $P --cond sham --body Panda --out $O/sham_prod
run sham_v3       $R --code $V --cond sham --body Panda --out $O/sham_v3
run sham_dev      $R --code $D --cond sham --body Panda --out $O/sham_dev
run sham_v3copy   COPY_GRIP_CMD=1 $R --code $V --cond sham --body Panda --out $O/sham_v3copy
run rec_prod      RECOLOR_ARM=1 $R --code $P --cond rec --body Panda --out $O/rec_prod
run rec_v3        RECOLOR_ARM=1 $R --code $V --cond rec --body Panda --out $O/rec_v3
run iiwa_prod     $R --code $P --cond iiwa --body IIWA --out $O/iiwa_prod
run iiwa_v3       $R --code $V --cond iiwa --body IIWA --out $O/iiwa_v3
run ur5e_prod     $R --code $P --cond ur5e --body UR5e --out $O/ur5e_prod
run ur5e_v3       $R --code $V --cond ur5e --body UR5e --out $O/ur5e_v3
run apps_prod     HIDE_ARM=1 $R --code $P --cond apps --body Panda --out $O/apps_prod
run apps_v3       HIDE_ARM=1 $R --code $V --cond apps --body Panda --out $O/apps_v3
run grp_dev       GRP_KP=100 $R --code $D --cond grpkp100 --body PandaR85 --out $O/grp_dev
run grp_v3        GRP_KP=100 $R --code $V --cond grpkp100 --body PandaR85 --out $O/grp_v3
wait
echo done $(date)
