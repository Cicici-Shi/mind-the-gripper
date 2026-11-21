#!/bin/bash
#SBATCH --job-name=zoo_dl_oft
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
# independent (single-suite) OpenVLA-OFT checkpoints into the vla_zoo HF cache (the published runs use $HOME/hf_cache)
export HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
for r in moojink/openvla-7b-oft-finetuned-libero-spatial moojink/openvla-7b-oft-finetuned-libero-object; do
  python -c "from huggingface_hub import snapshot_download as s; print(s(\"$r\"))"
done
du -sh $HF_HOME/hub/*
