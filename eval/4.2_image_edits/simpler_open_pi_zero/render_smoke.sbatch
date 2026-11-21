#!/bin/bash
#SBATCH -J b2s_render
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/render_%j.log
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
cd $PROJ_ROOT/bench2_simpler
hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,driver_version --format=csv
echo "== vk_diag nvidia ICD"; VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python vk_diag.py 2>&1 | tail -6
echo "== vulkan icd candidates"; ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -E "libGLX_nvidia|libEGL_nvidia|libvulkan"
echo "== try 1: default"
echo "skip smoke, done in job <job>"
if ! ls out/smoke_default/*.png >/dev/null 2>&1; then
  echo "== try 2: VK_ICD_FILENAMES=nvidia_icd.json"
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  python render_smoke.py out/smoke_icd 2>&1 | tail -40
fi
if true; then
  echo "== gripper probe"
  python gripper_probe.py out/gripper 2>&1 | tail -60
fi
ls -la out/*/
