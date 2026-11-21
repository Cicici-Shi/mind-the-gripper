#!/bin/bash
#SBATCH --job-name=v3_entry_pi05
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# Entry check: pi05_body_eval with EMBODY_DIR=embody_v3, 1 episode (task 0, episode 0) per new condition.
export TMPDIR=$HOME/tmp UV_CACHE_DIR=$HOME/.cache/uv XDG_CACHE_HOME=$HOME/.cache PIP_CACHE_DIR=$HOME/.cache/pip
export OPENPI_DATA_HOME=$HOME/.cache/openpi MUJOCO_GL=egl PYTHONFAULTHANDLER=1 HIDE_ARM=0 EMBODY_DIR=$PROJ_ROOT/embody_v3; unset DISPLAY
P=$PROJ_ROOT/pi05; O=$PROJ_ROOT/embody_v3/out/entry_pi05; mkdir -p $O
PORT=$((20000 + SLURM_JOB_ID % 20000))
source $HOME/miniforge3/etc/profile.d/conda.sh
cd $P/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pi05_libero --policy.dir=$P/ckpt/pi05_libero > $O/server.log 2>&1 &
SPID=$!
until python3 -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex((\"127.0.0.1\",$PORT)))" 2>/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; done
conda activate oftpi; cd $P
for spec in "p6j3 PandaL3" "k3 Kinova3" "rt PandaRT" "g_r Panda TWIN_RENDER=1 TWIN_BODY=PandaR85" \
            "grpkp100_tw PandaR85 GRP_KP=100 TWIN_RENDER=1 TWIN_CAMS=wrist" "grpkp100_ta PandaR85 GRP_KP=100 TWIN_RENDER=1 TWIN_CAMS=agent"; do
  set -- $spec; name=$1; body=$2; shift 2
  echo "=== $name BODY=$body $* $(date)"
  env "$@" python -X faulthandler pi05_body_eval.py --port $PORT --trials 1 --ep-offset 0 --tasks 0 --body $body --out $O/$name.jsonl > $O/$name.log 2>&1
  tail -c 600 $O/$name.jsonl; echo
done
kill $SPID; echo end $(date)
