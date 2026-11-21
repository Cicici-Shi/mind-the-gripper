#!/bin/bash
#SBATCH --job-name=toyprobe
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:59:00
#SBATCH --output=logs/%x_%j.log
# TOY1 probes (tools/probe_toy.py). Env: RUNS COND OUT TASKS DO MAXF (smoke: max frames).
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
T=$PROJ_ROOT/embody_toy; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) RUNS=$RUNS COND=$COND OUT=$OUT TASKS=${TASKS:-all} start=$(date)"
python -X faulthandler tools/probe_toy.py --runs $RUNS --cond $COND --out $OUT --do ${DO:-shift,parts} --tasks ${TASKS:-0,1,2,3,4,5,6,7,8,9} --max-frames ${MAXF:-0} --frame-rule ${RULE:-protocol}
echo "exit=$? end=$(date)"
