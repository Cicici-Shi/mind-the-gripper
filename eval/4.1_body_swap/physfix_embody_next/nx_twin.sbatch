#!/bin/bash
#SBATCH --job-name=nx_twin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
# embody_next twin-render check (no policy): CELL (zoo_cells.sh vars, e.g. g_tfc, sham_twin) on libero_goal + libero_10,
# replay of the SRC OFT actions, render 1 (natural scene draws on the eval's OffScreenRenderEnv path), twin IK residuals.
# Usage: sbatch tools/nx_twin.sbatch <cell> "<offsets>"
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
# or: sbatch tools/nx_twin.sbatch <tag> "<offsets>" <BODY> VAR=VAL ...   (cell given directly, e.g. the R140 GTfc analogue)
# SUITES (env, default "libero_goal libero_10") selects the suites.
CELL=$1; OFFS=${2:-"0 5 10 15"}; shift 2 || shift $#
N=$PROJ_ROOT/embody_next; O=$N/out/twin/$CELL; mkdir -p $O; cd $N
if [ $# -eq 0 ]; then set -- $(bash $PROJ_ROOT/vla_zoo/tools/zoo_cells.sh vars $CELL); fi
B=$1; shift
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date) cell=$CELL body=$B vars=$* body_env_sha=$(sha256sum embody/body_env.py | cut -c1-16)"
for s in ${SUITES:-libero_goal libero_10}; do for off in $OFFS; do
  env "$@" python tools/nx_check.py --code $N --suite $s --body $B --offset $off --modes replay --render 1 --out $O > $O/${s}_e$off.log 2>&1 &
  sleep 20
done; done
wait
grep -h DONE $O/*.log | wc -l
echo "end=$(date)"
