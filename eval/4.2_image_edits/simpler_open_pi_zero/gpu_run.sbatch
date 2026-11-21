#!/bin/bash
#SBATCH -J b2s_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/gpu_%j.log
# node23 上的 a100-40 是 A100-80 切出的 MIG 3g.40gb，Vulkan 起不来，所以排除 node23，只落在整卡 A100-PCIE-40GB 的 node15
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
B=$PROJ_ROOT/bench2_simpler; cd $B
export TRANSFORMERS_CACHE=$B/tf_cache VLA_LOG_DIR=$B/logs TOKENIZERS_PARALLELISM=false
hostname; nvidia-smi -L
echo "== gripper probe"
python gripper_probe.py out/gripper 2>&1 | grep -v "GLFW\|Continue without\|UserWarning\|logger.warn" | tail -60
echo "== baseline"
python baseline_pi0.py --task widowx_put_eggplant_in_basket --ckpt $B/ckpt/bridge_beta_step19296_2024-12-26_22-30_42.pt --n 10 --bf16 --out out/baseline_eggplant 2>&1 | grep -v "GLFW\|Continue without\|UserWarning\|logger.warn" | tail -40
ls -la out/*/ | head -40
