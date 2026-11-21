#!/bin/bash
#SBATCH --job-name=mipi_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:40:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi pre-freeze checks in ONE h200 allocation (the h200 quota is one card, the queue is long), run in order,
# each phase through its own script so every phase repeats the H200 / GL_RENDERER asserts:
#   1 tools/edit_equiv.sbatch   (frozen OFT hook vs openpi hook, same cached state, 256 edits byte for byte)
#   2 smoke pi05 spatial  C0..C8 task 0 ep 40 + reruns C0 / C5 / C8 (tag rerun), state probe
#   3 smoke pi0  spatial  C0..C8 task 0 ep 40 + reruns C0 / C5,                   state probe
#   4 smoke pi05 object   C0..C8 task 0 ep 40
#   5 tools/mi_pi_loadtest.sbatch (pi05 then pi0, 4 / 8 / 12 / 16 clients, 75 s windows)
#   6 tools/visibility_pi.py (Spatial cached states: expected-composite / not-vanished / edge / effective edit / wrist edge)
#   7 smoke pi0  object   C0..C8 task 0 ep $OBJ_OFF
#   8 tools/manip_pi.sbatch (protocol v2 2.6 manipulation checks, both suites, + t0 gate mode)
#   9 end-to-end S check runs (C0, 10 steps, --debug-state-transform; out/manip_pi/e2e_runs) + tools/state_e2e_pi.py
# OBJ_OFF (default 40; protocol v2 2.6 / 9 item 8 names the read Object key (0,0) -> pass OBJ_OFF=0)
# SMOKE_ROOT (default out/smoke_runs) separates smoke generations.
# Smoke keys are outside the formal keys (ep 20-29) and all clients are --blind. PHASES selects a subset.
set -u
T=$PROJ_ROOT/maskinfo_pi; cd $T
PHASES=${PHASES:-"1 2 3 4 5"}
SR=${SMOKE_ROOT:-$T/out/smoke_runs}
OO=${OBJ_OFF:-40}
SO="C0:$OO C1:$OO C2:$OO C3:$OO C4:$OO C5:$OO C6:$OO C7:$OO C8:$OO"
S="C0:40 C1:40 C2:40 C3:40 C4:40 C5:40 C6:40 C7:40 C8:40"
for ph in $PHASES; do
  echo "=== phase $ph $(date)"
  case $ph in
    1) bash tools/edit_equiv.sbatch ;;
    2) env MODEL=pi05 SUITE=spatial JOBS="$S C0:40:rerun C5:40:rerun C8:40:rerun" NTRIALS=1 TASKS=0 OUTROOT=$SR POST_PROBE=1 PORT_OFFSET=11 bash tools/mi_pi_run.sbatch ;;
    3) env MODEL=pi0 SUITE=spatial JOBS="$S C0:40:rerun C5:40:rerun" NTRIALS=1 TASKS=0 OUTROOT=$SR POST_PROBE=1 PORT_OFFSET=23 bash tools/mi_pi_run.sbatch ;;
    4) env MODEL=pi05 SUITE=object JOBS="$SO" NTRIALS=1 TASKS=0 OUTROOT=$SR PORT_OFFSET=37 bash tools/mi_pi_run.sbatch ;;
    5) bash tools/mi_pi_loadtest.sbatch ;;
    6) ( G="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; [[ "$G" == *H200* ]] || { echo "ASSERT card $G"; exit 4; }
         export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1; unset DISPLAY
         source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi; mkdir -p $T/out/visibility
         GLR=$(python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-); echo "GL_RENDERER=$GLR"; [[ "$GLR" == *H200* ]] || exit 5
         V=$T/out/visibility/vis_$SLURM_JOB_ID.jsonl
         python -X faulthandler tools/visibility_pi.py --out $V --keys ${VIS_KEYS:-0:40,1:40,5:40} --steps ${VIS_STEPS:-80} > $T/out/visibility/vis_$SLURM_JOB_ID.log 2>&1
         echo "visibility exit=$? tracebacks=$(grep -c Traceback $T/out/visibility/vis_$SLURM_JOB_ID.log)"
         python tools/visibility_pi.py --summary $V > $T/out/visibility/summary_$SLURM_JOB_ID.json; tail -1 $T/out/visibility/summary_$SLURM_JOB_ID.json ) ;;
    7) env MODEL=pi0 SUITE=object JOBS="$SO" NTRIALS=1 TASKS=0 OUTROOT=$SR PORT_OFFSET=41 bash tools/mi_pi_run.sbatch ;;
    8) bash tools/manip_pi.sbatch ;;
    9) E=$T/out/manip_pi/e2e_runs
       env MODEL=pi0 SUITE=spatial JOBS="C0:40" NTRIALS=1 TASKS=0 OUTROOT=$E CLIENT_ARGS="--max-steps 10" DEBUG_STATE_TRANSFORM=1 POST_PROBE=1 PORT_OFFSET=61 bash tools/mi_pi_run.sbatch
       env MODEL=pi05 SUITE=spatial JOBS="C0:40" NTRIALS=1 TASKS=0 OUTROOT=$E CLIENT_ARGS="--max-steps 10" DEBUG_STATE_TRANSFORM=1 POST_PROBE=1 PORT_OFFSET=67 bash tools/mi_pi_run.sbatch
       ( source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi
         { python -I tools/state_e2e_pi.py $E/pi0_spatial/C0/*.npz $E/pi05_spatial/C0/*.npz
           grep -h "^MI_STATE_PATH\|^MI_IMAGE_MASK" $T/logs/server/server_pi0_spatial_${SLURM_JOB_ID}_p61.log $T/logs/server/server_pi05_spatial_${SLURM_JOB_ID}_p67.log
           grep -h "^STATE_PROBE" $T/logs/mipi_pre*_${SLURM_JOB_ID}.log; } | tee $T/out/manip_pi/e2e_state.txt ) ;;
  esac
  echo "=== phase $ph exit=$? $(date)"
done
