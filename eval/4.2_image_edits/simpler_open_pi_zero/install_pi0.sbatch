#!/bin/bash
#SBATCH -J b2s_pi0inst
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/install_pi0_%j.log
export TMPDIR=$HOME/tmp
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
B=$PROJ_ROOT/bench2_simpler
df -h ~ | tail -1
# 1) 后台下载 Bridge-Beta 权重（公开，11.8GB）
pip install -q "huggingface_hub<0.27" 2>&1 | tail -1
mkdir -p $B/ckpt
( python - <<'PY'
import time; t=time.time()
from huggingface_hub import hf_hub_download
p = hf_hub_download("allenzren/open-pi-zero", "bridge_beta_step19296_2024-12-26_22-30_42.pt", local_dir="$PROJ_ROOT/bench2_simpler/ckpt")
print("DOWNLOADED", p, round(time.time()-t), "s", flush=True)
PY
) > $B/logs/dl_bridge_beta.log 2>&1 &
DL=$!
# 2) 换成 open-pi-zero 作者的 SimplerEnv fork（加了 proprio）
pip uninstall -y simpler_env mani_skill2_real2sim 2>&1 | tail -2
pip install -e $B/SimplerEnv_ozfork/ManiSkill2_real2sim 2>&1 | tail -1
pip install -e $B/SimplerEnv_ozfork 2>&1 | tail -1
# 3) open-pi-zero 推理依赖（不装训练用的 tfds/wandb；tensorflow 只为 tf.io.gfile 读统计量，用 cpu 版）
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
pip install transformers==4.47.1 hydra-core omegaconf einops bitsandbytes==0.45.0 pretty_errors sentencepiece "protobuf==3.20.3" tensorflow-cpu==2.15.0 "numpy==1.24.4" "setuptools<70" "opencv-python==4.10.0.84" 2>&1 | tail -2
pip install -e $B/open-pi-zero --no-deps 2>&1 | tail -1
python -c "import numpy, torch, transformers, tensorflow as tf, sapien, simpler_env, mani_skill2_real2sim; print('versions numpy', numpy.__version__, 'torch', torch.__version__, 'tf', tf.__version__, 'transformers', transformers.__version__); import simpler_env, os; print(simpler_env.__file__)"
# 4) 分词器
python $B/make_tokenizer.py ~/.cache/openpi/big_vision/paligemma_tokenizer.model $B/tf_cache/paligemma-3b-pt-224
pip check 2>&1 | head
wait $DL; cat $B/logs/dl_bridge_beta.log | tail -3; ls -la $B/ckpt; df -h ~ | tail -1
