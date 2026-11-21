#!/bin/bash
#SBATCH --job-name=nx_phys
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# embody_next full physics / scene check for one body (no policy, no render): libero_goal + libero_10, all 4 shards
# (EP_OFFSET 0/5/10/15 x 5 eps x 10 tasks = 200 keys per suite), modes replay (SRC OFT actions) and zero.
# Usage: sbatch --job-name=nx_phys_<tag> tools/nx_phys.sbatch <BODY> <tag> [VAR=VAL ...]
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
B=$1; TAG=$2; shift 2
N=$PROJ_ROOT/embody_next; O=$N/out/phys/$TAG; mkdir -p $O; cd $N
echo "node=$(hostname) start=$(date) body=$B vars=$* body_env_sha=$(sha256sum embody/body_env.py | cut -c1-16)"
for s in ${SUITES:-libero_goal libero_10}; do for off in 0 5 10 15; do
  env "$@" python tools/nx_check.py --code $N --suite $s --body $B --offset $off --modes replay,zero --zero-steps 200 --out $O > $O/${s}_e$off.log 2>&1 &
done; done
wait
grep -h DONE $O/*.log | wc -l
echo "end=$(date)"
