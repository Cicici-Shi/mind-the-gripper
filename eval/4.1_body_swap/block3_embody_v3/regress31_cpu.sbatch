#!/bin/bash
#SBATCH --job-name=v3_regr31
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# 2026-10-31 regression of the amendment-1 logging changes (contacts.first_grip_obj, align_info fields, G_R root offsets):
# old = backup_20261031 (code as of 10/27), new = embody_v3 with the contact log attached (--contacts 1). Physics only.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; B=$V/backup_20261031; O=$V/out/regress31; mkdir -p $O; cd $V
run() { local name=$1; shift; env "$@" > $O/$name.log 2>&1 & }
R="python tools/regress.py --eps 0-4"
run src_old    $R --code $B --cond src  --body SRC   --out $O/src_old
run src_new    $R --code $V --cond src  --body SRC   --out $O/src_new --contacts 1
run sham_old   $R --code $B --cond sham --body Panda --out $O/sham_old
run sham_new   $R --code $V --cond sham --body Panda --out $O/sham_new --contacts 1
run shamc_old  COPY_GRIP_CMD=1 $R --code $B --cond sham --body Panda --out $O/shamc_old
run shamc_new  COPY_GRIP_CMD=1 $R --code $V --cond sham --body Panda --out $O/shamc_new --contacts 1
run iiwa_old   $R --code $B --cond iiwa --body IIWA --out $O/iiwa_old
run iiwa_new   $R --code $V --cond iiwa --body IIWA --out $O/iiwa_new --contacts 1
run ur5e_old   $R --code $B --cond ur5e --body UR5e --out $O/ur5e_old
run ur5e_new   $R --code $V --cond ur5e --body UR5e --out $O/ur5e_new --contacts 1
run ur5nc_old  NO_FOREARM_CONTACT=1 COPY_GRIP_CMD=1 $R --code $B --cond ur5e --body UR5e --out $O/ur5nc_old --tasks 2,3
run ur5nc_new  NO_FOREARM_CONTACT=1 COPY_GRIP_CMD=1 $R --code $V --cond ur5e --body UR5e --out $O/ur5nc_new --tasks 2,3 --contacts 1
run grp_old    GRP_KP=100 $R --code $B --cond grpkp100 --body PandaR85 --out $O/grp_old
run grp_new    GRP_KP=100 $R --code $V --cond grpkp100 --body PandaR85 --out $O/grp_new --contacts 1
wait
echo done $(date)
