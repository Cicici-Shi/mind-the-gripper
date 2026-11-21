#!/bin/bash
#SBATCH -J b2c_stub
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/logs/stub_%j.log
export TMPDIR=$HOME/tmp PYTHONDONTWRITEBYTECODE=1 HF_HOME=$PROJ_ROOT/bench2_cogact/hf
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
python $PROJ_ROOT/bench2_cogact/make_llama_stub.py
