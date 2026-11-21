#!/bin/bash
#SBATCH --job-name=v3_replay
#SBATCH --partition=normal
#SBATCH --cpus-per-task=72
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Block-3 CPU diagnostics (ii)+(iii): closed-loop replay of the 194 successful SRC episodes on every
# body with embody_v3 (adds torque-saturation / stability fields). Panda, IIWA, UR5e, PandaR85 kp100
# also re-check the published closed-loop numbers ($PROJ_ROOT/replay/closed/*) episode by episode.
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 REPLAY_MODE=closed; unset DISPLAY
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
V=$PROJ_ROOT/embody_v3; O=$V/out/replay_closed; mkdir -p $O; cd $V
for spec in Panda:Panda IIWA:IIWA UR5e:UR5e PandaR85:PandaR85_kp100 PandaL3:PandaL3 Kinova3:Kinova3 PandaRT:PandaRT; do
  body=${spec%%:*}; name=${spec##*:}
  for t in 0 1 2 3 4 5 6 7 8 9; do
    extra=""; [ "$name" = PandaR85_kp100 ] && extra="GRP_KP=100"
    env TASKS=$t $extra python -m embody.replay $PROJ_ROOT/runs/src $body $O/${name}_t$t.jsonl > $O/${name}_t$t.log 2>&1 &
  done
done
wait
python tools/diag_track.py > $O/diag_track.txt 2>&1
echo done $(date)
