# 第二阶段（CogACT）每个 GPU 作业开头 source：环境、确认整卡、钉死 Vulkan 驱动、过滤噪声日志。由第一阶段 b2/jobs/common.sh 改写。
set -u
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TMPDIR=$HOME/tmp HF_HOME=$PROJ_ROOT/bench2_cogact/hf HF_HUB_OFFLINE=1
B2C=$PROJ_ROOT/bench2_cogact/b2c
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
# 起不来就干净地失败：B2C_NODE_FAIL 指向的文件（正式采集时是 formal_state/node_fail.tsv）记一行「节点、作业、原因」，
# 驱动脚本下次提交时把这些节点排除。这类失败不产生任何键的报错文件（没有跑到任何键）。
b2c_abort() { echo "ABORT: $*"; [ -n "${B2C_NODE_FAIL:-}" ] && printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}" "$*" >> "$B2C_NODE_FAIL"; exit "${B2C_ABORT_CODE:-3}"; }
hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current,memory.total --format=csv
# 卡型（协调方 2026-11-13 放宽）：B2C_ANY_GPU=1 只给不跑策略的检查作业用（键核对与卡型无关）；允许冻结清单列出的型号（A100-PCIE-40GB、H100 NVL），MIG 必须关闭（开了 MIG 的卡 Vulkan 起不来）
if nvidia-smi -L | grep -q MIG; then b2c_abort "MIG card"; fi
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -qv "Disabled\|N/A"; then b2c_abort "MIG mode not disabled"; fi
if [ "${B2C_ANY_GPU:-0}" != "1" ] && ! nvidia-smi -L | grep -q "A100-PCIE-40GB\|H100 NVL\|A100 80GB PCIe\|A100-SXM4-80GB"; then b2c_abort "GPU model not in allowed list (A100-PCIE-40GB, H100 NVL)"; fi
# 只给 Vulkan 加载器 nvidia 的 ICD（第一阶段协议 §4 硬门槛 4 的证据之一；第二阶段在 bench2_cogact 环境上重做了反向实验）。
# 值写进每条 npz（vk_icd_filenames、vk_driver_files），gate.py 硬门槛 4 核对。
B2_VK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
[ -f "$B2_VK_ICD" ] || b2c_abort "missing $B2_VK_ICD"
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES  # 新版加载器里 VK_DRIVER_FILES 优先于 VK_ICD_FILENAMES，先清掉
export VK_ICD_FILENAMES=$B2_VK_ICD
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
cd $B2C
NOISE="GLFW\|Continue without\|UserWarning\|logger.warn\|warnings.warn\|FutureWarning\|cpu_feature_guard\|AVX2 FMA\|DeprecationWarning\|logger.deprecation\|TensorRT\|cuda_dnn\|cuda_fft\|cuda_blas\|computation placer"
gpu_sampler() { while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> $1; sleep 5; done; }
# 渲染自检（所有跑策略的作业开头调用；协调方 2026-11-13：node37 上 SapienRenderer 会卡死）：90 秒内必须出帧，
# 否则以退出码 4 退出。只给 nvidia 驱动时建两个环境、取第 0 帧，必须与已知哈希逐位相同（第一阶段 icd 作业 <job>、
# 协调方各卡型探针、本阶段 check 作业都是这两个值）；起不来或不同就 b2c_abort，不跑任何键。
render_selfcheck() {
  local out; out=$(timeout 90 python icd_probe.py selfcheck 2>&1 | grep "\[icd\]")
  echo "$out"
  echo "$out" | grep -q "RESULT OK" || B2C_ABORT_CODE=4 b2c_abort "render selfcheck failed or no frame within 90 s"
  echo "$out" | grep -q "fa2ac84d9a68e8af281955b1fe6a964028198e2c5a3dd703ba14a1d8f3e9e44d" || b2c_abort "render selfcheck widowx frame0 differs"
  echo "$out" | grep -q "4e7330d264febb35c7077e6aec6bfbede01f102c47e53efee4ac09edc532737e" || b2c_abort "render selfcheck google frame0 differs"
}
