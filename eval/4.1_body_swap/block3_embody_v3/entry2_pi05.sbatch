#!/bin/bash
#SBATCH --job-name=v3_entry2_pi05
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Entry check 2 for pi0.5 (patched client copy pi05_body_eval_v3test.py), same cells as entry2_oft.
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYTHONFAULTHANDLER=1 HIDE_ARM=0 EMBODY_DIR=$PROJ_ROOT/embody_v3 COPY_GRIP_CMD=1; unset DISPLAY
P=$PROJ_ROOT/pi05; O=$PROJ_ROOT/embody_v3/out/entry2_pi05; mkdir -p $O
PORT=$((20000 + SLURM_JOB_ID % 20000))
source $HOME/miniforge3/etc/profile.d/conda.sh
cd $P/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero --policy.dir=$P/ckpt/pi05_libero > $O/server.log 2>&1 &
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex((\"127.0.0.1\",$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
conda activate oftpi; cd $P
while read suite note body extra; do
  [ -z "$suite" ] && continue
  b=""; [ "$body" != SRC ] && b="--body $body"
  echo "=== $suite $note $body $extra $(date)"
  env $extra python -X faulthandler pi05_body_eval_v3test.py --port $PORT --suite $suite --trials 1 --ep-offset 0 --tasks 0 $b --out $O/${suite}_$note.jsonl > $O/${suite}_$note.log 2>&1
  tail -c 300 $O/${suite}_$note.jsonl; echo
done <<LIST
libero_spatial g_r_sunk Panda TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk
libero_spatial ur5e_nc UR5e NO_FOREARM_CONTACT=1
libero_spatial rt2 PandaRT2 X=0
libero_object src SRC X=0
libero_object s Panda X=0
libero_object t0 Panda TWIN_RENDER=1
libero_object ur5e UR5e X=0
libero_object k3 Kinova3 X=0
libero_object g PandaR85 GRP_KP=100
libero_object g_t PandaR85 GRP_KP=100 TWIN_RENDER=1
LIST
kill $SPID; echo end $(date)
