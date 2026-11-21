#!/bin/bash
#SBATCH --job-name=v3_chk
#SBATCH --partition=normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 REPLAY_MODE=closed; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/replay_closed_cg; mkdir -p $O; cd $V
for b in Panda IIWA; do for t in 0 1 2 3 4 5 6 7 8 9; do
  env TASKS=$t COPY_GRIP_CMD=1 python -m embody.replay $PROJ_ROOT/runs/src $b $O/${b}_t$t.jsonl > $O/${b}_t$t.log 2>&1 &
done; done
python tools/rt_gap.py > $O/rt_gap.txt 2>&1
wait; echo done
