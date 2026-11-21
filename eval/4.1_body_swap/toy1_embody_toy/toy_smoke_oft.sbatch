#!/bin/bash
#SBATCH --job-name=toy_smk
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=24
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# TOY1 OFT entry smoke: each new cell 2 episodes (tasks 0,1, episode 0) through oft/run_libero_eval_toy.py, all cells as
# parallel processes on one GPU; then the fingertip probe on the fingers-cell dumps. Not a result: dumps go to
# out/smoke_oft_<job>/ and are not used later.
export TMPDIR=$HOME/tmp HF_HOME=$HOME/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false ORIGINAL_ONLY=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
T=$PROJ_ROOT/embody_toy; export EMBODY_DIR=$T PYTHONPATH=$PROJ_ROOT/openvla-oft
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft; cd $T
O=$T/out/smoke_oft_$SLURM_JOB_ID; mkdir -p $O $T/logs/eval
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "node=$(hostname) gpu=$NUS_GPU cpus=$SLURM_CPUS_PER_TASK start=$(date)"
B="BODY=PandaR85 GRP_KP=100 COPY_GRIP_CMD=1 TOY_PIXSTATS=1 QUERY_DUMP=1 MAX_TASKS=2"
for spec in "G_camback|$B TOY_WRIST_CAM=back" "GT_camphys|$B TWIN_RENDER=1 TOY_WRIST_CAM=phys" \
            "fingers|$B TWIN_RENDER=1 TOY_PARTS=fingers" "palm|$B TWIN_RENDER=1 TOY_PARTS=palm" "hand|$B TWIN_RENDER=1 TOY_PARTS=hand"; do
  IFS='|' read n v <<< "$spec"
  env $v DUMP_DIR=$O/runs/$n EP_OFFSET=0 python -X faulthandler oft/run_libero_eval_toy.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --task_suite_name libero_spatial --num_trials_per_task 1 --run_id_note smoke_$n --local_log_dir $T/logs/eval > $O/$n.log 2>&1 &
  sleep 20
done
( while pgrep -u $USER -f run_libero_eval_toy >/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1)"; sleep 30; done ) > $O/gpu.tsv &
wait
for n in G_camback GT_camphys fingers palm hand; do echo "$n: episodes $(grep -c 'Starting episode' $O/$n.log) tracebacks $(grep -c Traceback $O/$n.log) dumps $(ls $O/runs/$n 2>/dev/null | wc -l)"; grep -h "TOY1\|BODY_ALIGN" $O/$n.log | head -2; done
env BODY=PandaR85 GRP_KP=100 COPY_GRIP_CMD=1 TWIN_RENDER=1 TOY_PARTS=fingers \
  python -X faulthandler tools/tip_probe.py --runs $O/runs --cond fingers --frames preclose:3 --d 5 --out $O/probe_fingers > $O/probe.log 2>&1
echo "probe exit=$? end=$(date)"
