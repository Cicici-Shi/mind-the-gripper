#!/bin/bash
#SBATCH --job-name=v3_chk2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# Does the published prod replay.py reproduce its own 10-26 closed-loop Panda numbers today?
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 REPLAY_MODE=closed; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
O=$PROJ_ROOT/embody_v3/out/replay_closed_prodnow; mkdir -p $O; cd $PROJ_ROOT/embody
for t in 0 1 2 3 4 5 6 7 8 9; do env TASKS=$t python -m embody.replay $PROJ_ROOT/runs/src Panda $O/Panda_t$t.jsonl > $O/Panda_t$t.log 2>&1 & done
wait; echo done
