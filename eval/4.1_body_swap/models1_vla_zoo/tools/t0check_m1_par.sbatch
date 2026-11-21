#!/bin/bash
#SBATCH --job-name=m1_t0chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# models1 7.2 step-0 consistency, one suite, 5 processes of 2 tasks each (shard-0 keys ep 0-4). Each process seeds 7
# and starts at its first task, so fixture placement may differ from the policy runs for tasks > 0 of a group; the
# check compares cells on the SAME state inside one process, which is all it needs.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
export EMBODY_DIR=$PROJ_ROOT/embody_toy
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/t0check; mkdir -p $O; cd $Z
for g in 0,1 2,3 4,5 6,7 8,9; do
  python tools/t0check_m1.py --suite $SUITE --tasks $g --out $O/t0_${SUITE}_${g/,/}.jsonl > $O/t0_${SUITE}_${g/,/}.log 2>&1 &
done
wait; echo done $(date)
