#!/bin/bash
#SBATCH -J b2c_gpu2
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --nice=200
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/gpu_%j.log
# 接手后的第二个 GPU 作业：WidowX 夹爪探测（spoon）+ WidowX 两任务基线各 10 回合
# 用法：sbatch gpu_run2.sbatch <probe_tasks 或 none> <probe_out> <tasks> <n> <outname>
nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,memory.total --format=csv
export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1 HF_HOME=$PROJ_ROOT/bench2_cogact/hf TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
R=$PROJ_ROOT/bench2_cogact; cd $R
python -c "import torch; sd=torch.load('$R/ckpt/CogACT-Base/checkpoints/CogACT-Base.pt', map_location='cpu', mmap=True)['model']; print('CKPT top keys', sorted(sd.keys()))" 2>&1 | grep -v -E "cuda_|TensorRT|cpu_feature|AVX2|NVML"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R/logs/smi_$SLURM_JOB_ID.txt; sleep 5; done ) &
SMI=$!
if [ "$1" != "none" ]; then python $R/gripper_probe_cogact.py $R/out/$2 $1 2>&1 | grep -v -E "GLFW|DISPLAY"; fi
python $R/baseline_cogact.py $3 $4 $R/out/$5 2>&1 | grep -v -E "^[0-9]+ \{|GLFW|DISPLAY"
kill $SMI
echo "nvidia-smi peak MiB: $(sort -n $R/logs/smi_$SLURM_JOB_ID.txt | tail -1)"
