#!/bin/bash
#SBATCH -J b2c_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --nice=200
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/gpu_%j.log
# 用法：sbatch gpu_run.sbatch <probe:0/1> <tasks> <n> <outname>
nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,memory.total --format=csv
export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1 HF_HOME=$PROJ_ROOT/bench2_cogact/hf TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
R=$PROJ_ROOT/bench2_cogact; cd $R
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R/logs/smi_$SLURM_JOB_ID.txt; sleep 5; done ) &
SMI=$!
if [ "$1" = "1" ]; then python $R/gripper_probe_cogact.py $R/out/gripper 2>&1 | grep -v -E "GLFW|DISPLAY"; fi
python $R/baseline_cogact.py $2 $3 $R/out/$4 2>&1 | grep -v -E "^[0-9]+ \{|GLFW|DISPLAY"
kill $SMI
echo "nvidia-smi peak MiB: $(sort -n $R/logs/smi_$SLURM_JOB_ID.txt | tail -1)"
