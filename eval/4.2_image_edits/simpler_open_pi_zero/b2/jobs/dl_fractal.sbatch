#!/bin/bash
#SBATCH -J b2_dlfrac
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/dl_fractal_%j.log
export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bench2_simpler
python - <<'PY'
import time; t=time.time()
from huggingface_hub import hf_hub_download
p = hf_hub_download("allenzren/open-pi-zero", "fractal_beta_step29576_2024-12-29_13-10_42.pt", local_dir="$PROJ_ROOT/bench2_simpler/b2/ckpt")
print("DOWNLOADED", p, round(time.time()-t), "s", flush=True)
PY
ls -la $PROJ_ROOT/bench2_simpler/b2/ckpt
