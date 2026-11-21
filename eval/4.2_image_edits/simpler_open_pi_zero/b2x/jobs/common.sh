# 每个 GPU 作业开头 source：环境、确认整卡、过滤噪声日志（b2x：补充协议；与 b2 的差别只有目录）
set -u
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
B=$PROJ_ROOT/bench2_simpler; B2=$B/b2x   # 补充协议的代码目录（b2 已冻结只读）
export TRANSFORMERS_CACHE=$B/tf_cache VLA_LOG_DIR=$B2/logs
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,memory.total --format=csv
if nvidia-smi -L | grep -q MIG; then echo "MIG card, abort"; exit 1; fi
if ! nvidia-smi -L | grep -q "A100-PCIE-40GB\|A100-SXM4-40GB"; then echo "not a full A100-40, abort"; exit 1; fi
# 只给 Vulkan 加载器 nvidia 的 ICD（协议硬门槛 4 的证据之一；icd_probe 作业验证过只给它能正常出图）。
# 值写进每条 npz（vk_icd_filenames），gate.py 硬门槛 4 核对。
B2_VK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
[ -f "$B2_VK_ICD" ] || { echo "missing $B2_VK_ICD, abort"; exit 1; }
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES  # 新版加载器里 VK_DRIVER_FILES 优先于 VK_ICD_FILENAMES，先清掉
export VK_ICD_FILENAMES=$B2_VK_ICD
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
cd $B2
NOISE="GLFW\|Continue without\|UserWarning\|logger.warn\|warnings.warn\|FutureWarning\|cpu_feature_guard\|AVX2 FMA\|DeprecationWarning\|logger.deprecation"
gpu_sampler() { while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> $1; sleep 5; done; }
# 只等数组 P 里的进程（不能用裸 wait：后台的 gpu_sampler 永不退出），并报告每个进程的退出码
waitall() { for p in "${P[@]}"; do wait $p; echo "pid $p exit $?"; done; }
