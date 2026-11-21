#!/bin/bash
#SBATCH -J b2c_check
#SBATCH -p gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/check_%j.log
# 第二阶段冻结前：1 硬门槛 4 的反向实验（bench2_cogact 环境，三种 ICD 设置；放在 nv 小卡上，A100-40 上已由第一阶段做过）；2 键核对
# （初始状态与第 0 帧和卡型无关，协调方的探针已示 H100 NVL 与 A100-40 第 0 帧逐位相同）（keycheck.py，四块全部正式键与校准键）。不跑模型。
export B2C_ANY_GPU=1  # 不跑策略：键核对与卡型无关；ICD 反向实验顺带在这种卡上再做一遍
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
echo "== icd.d"; ls -la /usr/share/vulkan/icd.d/
echo "== 1 ICD 反向实验"
(unset VK_ICD_FILENAMES; timeout 300 python icd_probe.py default 2>&1 | grep -v "$NOISE") | tee logs/check_${SLURM_JOB_ID}_icd_default.txt
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json timeout 300 python icd_probe.py lvp_only 2>&1 | grep -v "$NOISE" | tee logs/check_${SLURM_JOB_ID}_icd_lvp.txt
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json timeout 300 python icd_probe.py nvidia_only 2>&1 | grep -v "$NOISE" | tee logs/check_${SLURM_JOB_ID}_icd_nvidia.txt
D=$(grep "sha256" logs/check_${SLURM_JOB_ID}_icd_default.txt | awk '{print $NF}' | tr '\n' ' ')
NV=$(grep "sha256" logs/check_${SLURM_JOB_ID}_icd_nvidia.txt | awk '{print $NF}' | tr '\n' ' ')
echo "default sha: $D"; echo "nvidia  sha: $NV"
if grep -q "RESULT OK" logs/check_${SLURM_JOB_ID}_icd_nvidia.txt && [ "$D" = "$NV" ]; then echo "ICD_CHECK nvidia-only == default"; else echo "ICD_CHECK FAILED"; fi
grep -q "RESULT FAIL" logs/check_${SLURM_JOB_ID}_icd_lvp.txt && echo "ICD_CHECK lvp-only cannot start (expected)" || echo "ICD_CHECK lvp-only STARTED (unexpected)"
echo "== 2 键核对"
python keycheck.py --out keycheck.json 2>&1 | grep -v "$NOISE\|svulkan"
