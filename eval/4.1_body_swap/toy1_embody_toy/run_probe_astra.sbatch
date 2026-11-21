#!/bin/bash
#SBATCH --job-name=toyprobe
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# TOY1 probes under protocol_toy1_astra_probe_rule.md: grpkp100 ep 6-11 per task, frames t_c-5..t_c-1, both appearances
# + parts c0..c6. Task groups run as parallel processes on one GPU (no RNG dependence between tasks). Env: TASKGROUPS, OUT.
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/embody_toy; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
TASKGROUPS=${TASKGROUPS:-"0,1,2 3,4,5 6,7 8,9"}; OUT=${OUT:-$T/out/probe_astra}
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) TASKGROUPS=$TASKGROUPS OUT=$OUT MAXF=${MAXF:-0} start=$(date)"
i=0; for g in $TASKGROUPS; do
  python -X faulthandler tools/probe_toy.py --runs $PROJ_ROOT/runs --cond grpkp100 --eps 6-11 --frame-rule astra \
    --tasks $g --out $OUT/p$i --max-frames ${MAXF:-0} --save-png 3 > $T/logs/probe_${SLURM_JOB_ID}_p$i.log 2>&1 &
  i=$((i+1)); sleep 15; done
( while pgrep -u $USER -f "probe_toy.py" >/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1)"; sleep 60; done ) > $T/logs/probe_${SLURM_JOB_ID}_gpu.tsv &
wait
for j in $(seq 0 $((i-1))); do echo "p$j: $(grep -c '"frames"' $T/logs/probe_${SLURM_JOB_ID}_p$j.log) episodes, tracebacks $(grep -c Traceback $T/logs/probe_${SLURM_JOB_ID}_p$j.log)"; done
echo "end=$(date)"
