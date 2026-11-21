#!/bin/bash
#SBATCH --job-name=v3_replay2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=72
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Block-3 CPU gates (protocol_block3.md 4.3) and R7 contact predictions: closed-loop replay of the 194
# successful SRC episodes with online contact logging, all with COPY_GRIP_CMD=1. One node for all bodies.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 REPLAY_MODE=closed COPY_GRIP_CMD=1; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/replay_b3; mkdir -p $O; cd $V
for spec in Panda:Panda IIWA:IIWA UR5e:UR5e PandaL3:PandaL3 Kinova3:Kinova3 PandaRT2:PandaRT2 UR5e:UR5e_nc; do
  body=${spec%%:*}; name=${spec##*:}
  for t in 0 1 2 3 4 5 6 7 8 9; do
    extra="X=0"; [ "$name" = UR5e_nc ] && extra="NO_FOREARM_CONTACT=1"
    env TASKS=$t $extra python -m embody.replay $PROJ_ROOT/runs/src $body $O/${name}_t$t.jsonl > $O/${name}_t$t.log 2>&1 &
  done
done
wait; echo node=$(hostname) $(lscpu | grep "Model name"); echo done $(date)
