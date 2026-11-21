#!/bin/bash
#SBATCH --job-name=v3_chk3
#SBATCH --partition=normal
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# prod replay.py vs embody_v3 replay.py (diagnostic fields added) on the SAME node: closed-loop Panda.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 REPLAY_MODE=closed; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
O=$PROJ_ROOT/embody_v3/out/replay_samenode; mkdir -p $O/prod $O/v3
for t in 0 1 2 3 4 5 6 7 8 9; do
 (cd $PROJ_ROOT/embody && TASKS=$t python -m embody.replay $PROJ_ROOT/runs/src Panda $O/prod/Panda_t$t.jsonl > $O/prod/$t.log 2>&1) &
 (cd $PROJ_ROOT/embody_v3 && TASKS=$t python -m embody.replay $PROJ_ROOT/runs/src Panda $O/v3/Panda_t$t.jsonl > $O/v3/$t.log 2>&1) &
done; wait; echo node=$(hostname) $(lscpu | grep "Model name")
