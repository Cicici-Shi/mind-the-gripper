#!/bin/bash
#SBATCH --job-name=cond_g_r_e0
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:59:00
#SBATCH --output=logs/cond_%x_%j.log
# Block 3 G_R shard 0 (amendment 1, section 6): in the same job, first the ruling-2 (i)(ii) step-0 check
# (tools/gr_t0check.py, same node and GPU), then the OFT eval exactly as ~/run_cond.sbatch with NOTE=g_r.
# Root-body offsets (ruling 3.1) are written into every dump (twin_root_off_mm) by run_libero_eval.py.
export TMPDIR=$HOME/tmp; export HF_HOME=$HOME/hf_cache
export MUJOCO_GL=egl; unset DISPLAY; export TOKENIZERS_PARALLELISM=false
export ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1
V=$PROJ_ROOT/embody_v3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate oft
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) NOTE=$NOTE BODY=${BODY:-SRC} TRIALS=$TRIALS EP_OFFSET=${EP_OFFSET:-0} start=$(date)"
mkdir -p $V/out/gr_t0
( cd $V && OPENBLAS_NUM_THREADS=1 python -X faulthandler tools/gr_t0check.py --offsets ${EP_OFFSET:-0} \
    --out $V/out/gr_t0/oft_${SLURM_JOB_ID}.jsonl --save-png $V/out/gr_t0/png_${SLURM_JOB_ID} ) > $V/out/gr_t0/oft_${SLURM_JOB_ID}.log 2>&1
echo "gr_t0check exit=$? $(date)"
export DUMP_DIR=$PROJ_ROOT/runs/$NOTE
cd $PROJ_ROOT/openvla-oft
python -X faulthandler experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --task_suite_name libero_spatial \
  --num_trials_per_task $TRIALS \
  --run_id_note $NOTE
echo "exit=$? end=$(date)"
