#!/bin/bash
#SBATCH --job-name=mipi_manip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi manipulation checks (protocol_maskinfo_pi.md v2 2.6) on cached states, h200 only, with a policy server
# occupying the card (2.6 item 12: render determinism is checked while the server holds its memory). No policy is
# queried by the checks, no outcome is read. Groups (one process each): GROUPS_T="suite:tasks:offset[:t0] ...";
# default = snapshot mode Spatial (S_hook ep 40-49 actions) offsets 40,45 and Object (obj_s ep 0-9) offsets 0,5, all
# ten tasks, first key of each shard (2 keys / task), sheets for the first 3 keys; t0 mode (gate item 3) on the formal
# keys ep 20-29 with borrowed actions (spatial +20, object -20), 24 steps, no sheet.
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 OPENPI_DATA_HOME=$HOME/.cache/openpi HIDE_ARM=0; unset DISPLAY
T=$PROJ_ROOT/maskinfo_pi; cd $T
export NUS_GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, /|/')"
[[ "$NUS_GPU" == *H200* ]] || { echo "ASSERT card is not H200: $NUS_GPU"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh
GLR=$(conda activate oftpi; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
echo "GL_RENDERER=$GLR"; [[ "$GLR" == *H200* ]] || { echo "ASSERT GL_RENDERER is not H200: $GLR"; exit 5; }
eval "$(bash $T/tools/mi_pi_params.sh)"
O=$T/out/manip_pi/run_$SLURM_JOB_ID; mkdir -p $O
GROUPS_T=${GROUPS_T:-"spatial:0,1,2,3,4,5,6,7,8,9:40 spatial:0,1,2,3,4,5,6,7,8,9:45 object:0,1,2,3,4,5,6,7,8,9:0 object:0,1,2,3,4,5,6,7,8,9:5 spatial:0,1,2,3,4:20:t0 spatial:5,6,7,8,9:20:t0 spatial:0,1,2,3,4:25:t0 spatial:5,6,7,8,9:25:t0 object:0,1,2,3,4:20:t0 object:5,6,7,8,9:20:t0 object:0,1,2,3,4:25:t0 object:5,6,7,8,9:25:t0"}
echo "node=$(hostname) gpu=$NUS_GPU GROUPS_T=$GROUPS_T lag=$MI_LAG_STEPS width=$MI_OUTLINE_W start=$(date)"
sha256sum embody/maskinfo.py embody/maskrepl.py embody/maskinfo_pi.py tools/manip_maskinfo_pi.py pi/zoo_client_mi.py
PORT=$((20000 + (SLURM_JOB_ID + 53) % 20000))
( cd $PROJ_ROOT/pi05/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC exec .venv/bin/python $T/pi/serve_policy_mi.py \
    --config pi05_libero --dir $PROJ_ROOT/pi05/ckpt/pi05_libero --port $PORT --state-log $O/state_unused.jsonl ) > $O/server.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null 2>&1; do sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; exit 1; }; done
echo "server occupying the card: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
conda activate oftpi
i=0; PIDS=()
for it in $GROUPS_T; do IFS=: read su tk off mode <<< "$it"
  X="--png-keys 3"; [ "$mode" = t0 ] && X="--t0 --only-first 5 --max-steps 24 --png-keys 0"
  case $i in 0|2) ;; *) X="$X --png-keys 0" ;; esac   # sheets: first Spatial and first Object group
  python -X faulthandler tools/manip_maskinfo_pi.py --suite $su --tasks $tk --offsets $off --lag $MI_LAG_STEPS --width $MI_OUTLINE_W \
    --protect $MI_PROTECT --out $O/m_${i}${mode:+_t0}.jsonl --png-dir $O/png $X > $O/p${i}.log 2>&1 &
  PIDS+=($!); i=$((i+1)); sleep 3
done
wait "${PIDS[@]}"
for f in $O/p*.log; do echo "$f: keys $(grep -c '^{' $f) tracebacks $(grep -c Traceback $f) asserts $(grep -c 'AssertionError\|LAG_ASSERT' $f) done $(grep -c 'MANIP_MI done' $f)"; done
python tools/manip_summary_pi.py $(ls $O/m_*.jsonl | grep -v _t0) --json $O/summary.json > $O/summary.txt 2>&1; echo "summary exit=$?"
python3 -c "import json; k=json.load(open('$O/summary.json'))['keys']; print('RULE_LAG', json.dumps(k['rule_lag'])); print('VISIBILITY_HARD_OK', k['visibility_hard_ok']); print('physics_max', k['physics_max'], 'gpus', k['gpus'])"
kill $SPID 2>/dev/null; echo "end=$(date)"
