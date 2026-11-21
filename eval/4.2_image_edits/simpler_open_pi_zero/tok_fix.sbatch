#!/bin/bash
#SBATCH -J b2s_tok
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/logs/tok_fix_%j.log
# 新 sentencepiece 的 pb2 要 protobuf>=5，而 tensorflow 2.15 / open-pi-zero 钉 protobuf 3.20.3，所以降 sentencepiece
export TMPDIR=$HOME/tmp
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
B=$PROJ_ROOT/bench2_simpler
pip install "sentencepiece==0.1.99" 2>&1 | tail -1
rm -rf $B/tf_cache/paligemma-3b-pt-224
python $B/make_tokenizer.py ~/.cache/openpi/big_vision/paligemma_tokenizer.model $B/tf_cache/paligemma-3b-pt-224 2>&1 | tail -5
ls $B/tf_cache/paligemma-3b-pt-224
cd $B/open-pi-zero && TRANSFORMERS_CACHE=$B/tf_cache VLA_LOG_DIR=$B/logs python -c "
from omegaconf import OmegaConf; import hydra
from src.model.vla.pizero import PiZeroInference
cfg = OmegaConf.load('config/eval/bridge.yaml')
ad = hydra.utils.instantiate(cfg.env.adapter)
print('adapter ok, image_token_id', ad.processor.image_token_id, 'cfg image_token_index', cfg.image_token_index)
"
