#!/bin/bash
#SBATCH --job-name=mipi_equiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%j.log
# maskinfo-pi: same cached state, frozen OFT hook (env oft) vs openpi hook (env oftpi): 256 edited observations byte
# for byte (tools/edit_equiv_pi.py). h200 + GL_RENDERER H200 asserted. No policy, no outcome.
set -u
export TMPDIR=$HOME/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 HIDE_ARM=0
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; unset DISPLAY
T=$PROJ_ROOT/maskinfo_pi; cd $T; O=$T/out/edit_equiv; mkdir -p $O
G="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; [[ "$G" == *H200* ]] || { echo "ASSERT card $G"; exit 4; }
source $HOME/miniforge3/etc/profile.d/conda.sh
for envn in oft oftpi; do
  GLR=$(conda activate $envn; python tools/gl_renderer.py 2>&1 | grep "^GL_RENDERER" | cut -d" " -f2-)
  echo "GL_RENDERER[$envn]=$GLR"; [[ "$GLR" == *H200* ]] || { echo "ASSERT GL_RENDERER $GLR"; exit 5; }
done
KEYS=${KEYS:-0:40,1:40}; STEPS=${STEPS:-60}
( conda activate oft; PYTHONPATH=$PROJ_ROOT/openvla-oft python -X faulthandler tools/edit_equiv_pi.py --mode frozen --keys $KEYS --steps $STEPS --out $O/frozen_$SLURM_JOB_ID.jsonl ) > $O/frozen_$SLURM_JOB_ID.log 2>&1 &
P1=$!
( conda activate oftpi; python -X faulthandler tools/edit_equiv_pi.py --mode pi --keys $KEYS --steps $STEPS --out $O/pi_$SLURM_JOB_ID.jsonl ) > $O/pi_$SLURM_JOB_ID.log 2>&1 &
P2=$!
wait $P1; echo "frozen exit=$? tracebacks=$(grep -c Traceback $O/frozen_$SLURM_JOB_ID.log)"
wait $P2; echo "pi exit=$? tracebacks=$(grep -c Traceback $O/pi_$SLURM_JOB_ID.log)"
( conda activate oftpi; python tools/edit_equiv_pi.py --mode compare --a $O/frozen_$SLURM_JOB_ID.jsonl --b $O/pi_$SLURM_JOB_ID.jsonl )
echo "end=$(date)"
