#!/bin/bash
#SBATCH -J b2_icdcheck
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/icdcheck_%j.log
# 协议 v1.1 冻结前（接手方）：
#  1 硬门槛 4 的反向实验：默认 / 只给 lavapipe / 只给 nvidia 三种 ICD 设置下渲染能否起来、第 0 帧是否相同；
#  2 只给 nvidia 能出图且与默认逐位相同时，用 common.sh（导出 nvidia ICD）重跑两个冒烟键的 C0（B1、F1，盲成败），
#    第 0 帧与 out_smoke 里已有那帧逐字节比对；
#  3 键核查：四块全部正式键与校准键（probe_keys.py）。
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
B2=$PROJ_ROOT/bench2_simpler/b2; cd $B2
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,memory.total --format=csv
if nvidia-smi -L | grep -q MIG; then echo "MIG card, abort"; exit 1; fi
echo "== icd.d on compute node"; ls -la /usr/share/vulkan/icd.d/; echo "-- nvidia_icd.json"; cat /usr/share/vulkan/icd.d/nvidia_icd.json; echo "-- lvp_icd.json"; cat /usr/share/vulkan/icd.d/lvp_icd.json
which vulkaninfo && { for v in "" /usr/share/vulkan/icd.d/lvp_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do echo "-- vulkaninfo --summary VK_ICD_FILENAMES=$v"; if [ -z "$v" ]; then vulkaninfo --summary 2>&1 | grep -i "deviceName\|deviceType\|driverName\|ERROR" | head; else VK_ICD_FILENAMES=$v vulkaninfo --summary 2>&1 | grep -i "deviceName\|deviceType\|driverName\|ERROR" | head; fi; done; }
NOISE="GLFW\|Continue without\|UserWarning\|logger.warn\|warnings.warn\|FutureWarning\|cpu_feature_guard\|AVX2 FMA\|DeprecationWarning\|logger.deprecation"
echo "== 1 ICD 反向实验"
(unset VK_ICD_FILENAMES; timeout 300 python icd_probe.py default 2>&1 | grep -v "$NOISE") | tee logs/icdcheck_${SLURM_JOB_ID}_default.txt
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json timeout 300 python icd_probe.py lvp_only 2>&1 | grep -v "$NOISE" | tee logs/icdcheck_${SLURM_JOB_ID}_lvp.txt
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json timeout 300 python icd_probe.py nvidia_only 2>&1 | grep -v "$NOISE" | tee logs/icdcheck_${SLURM_JOB_ID}_nvidia.txt
D=$(grep "sha256" logs/icdcheck_${SLURM_JOB_ID}_default.txt | awk '{print $NF}' | tr '\n' ' ')
NV=$(grep "sha256" logs/icdcheck_${SLURM_JOB_ID}_nvidia.txt | awk '{print $NF}' | tr '\n' ' ')
echo "default sha: $D"; echo "nvidia  sha: $NV"
if ! grep -q "RESULT OK" logs/icdcheck_${SLURM_JOB_ID}_nvidia.txt || [ "$D" != "$NV" ]; then
  echo "NVIDIA-ONLY ICD FAILED OR DIFFERS FROM DEFAULT; stop before step 2"; exit 2
fi
echo "nvidia-only == default (bit-identical first frames)"
echo "== 2 重跑冒烟键 C0（导出 nvidia ICD）"
source $B2/jobs/common.sh
rm -rf out_icdcheck; P=()
for b in B1 F1; do
  python eval_views.py --block $b --cell C0 --keys keysets/smoke_$b.json --out out_icdcheck --save-frames 0 --blind-success \
    > logs/icdcheck_${SLURM_JOB_ID}_${b}_C0.log 2>&1 &
  P+=($!); sleep 10
done; waitall
grep -h "done\|ERROR" logs/icdcheck_${SLURM_JOB_ID}_*_C0.log
python - <<'PY'
import glob, hashlib, os, json
import numpy as np, imageio
for p in sorted(glob.glob("out_icdcheck/*/C0/*.npz")):
    z = np.load(p)
    b, k = str(z["block"]), str(z["key"])
    old_png = f"out_smoke/{b}/C0/frames/{k}_q000_full.png"
    new_png = f"out_icdcheck/{b}/C0/frames/{k}_q000_full.png"
    old = np.ascontiguousarray(imageio.imread(old_png))
    new = np.ascontiguousarray(imageio.imread(new_png))
    h_old = hashlib.sha256(old.tobytes()).hexdigest()
    print(f"[cmp] {k} vk={str(z['vk_icd_filenames'])} frame0_sha256(new npz)={str(z['frame0_sha256'])}")
    print(f"[cmp] {k} sha256(old smoke png pixels)={h_old} bytes_equal_new_vs_old_png={np.array_equal(new, old)} "
          f"npz_hash_equals_old={str(z['frame0_sha256']) == h_old} old_png_mtime={os.path.getmtime(old_png):.0f}")
PY
echo "== 3 键核查"
python probe_keys.py --out keyprobe_v11.json 2>&1 | grep -v "$NOISE\|svulkan"
