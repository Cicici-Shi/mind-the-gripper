#!/bin/bash
#SBATCH --job-name=ovr_smoke
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/ovr_smoke_%j.log
# Fable add1 4.4 smoke: SRC (native Panda) cell, ONE key, against a server address given in SRV_ADDR (a file
# "host port jobid gpu"); plus a no-policy field self-test of the g_tw_fc recorder (5 DUMMY steps, no server).
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONFAULTHANDLER=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset DISPLAY
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/ovla_rerun/smoke; mkdir -p $O
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
read -r HOST PORT SJ SGPU < $SRV_ADDR
export NUS_SRV_HOST=$HOST NUS_SRV_GPU="$SGPU"
echo "node=$(hostname) gpu=$NUS_GPU server=$HOST:$PORT (job $SJ, $SGPU) start=$(date)"
sha256sum $Z/tools/ovla_qpos_rerun.py $Z/tools/ovla_rerun_compare.py $Z/common/zoo_client.py $Z/common/m1_fastsync.py $Z/tools/zoo_cells.sh $PROJ_ROOT/embody_toy/embody/body_env.py
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi; cd $HOME/tmp
python -u $Z/tools/ovla_qpos_rerun.py selftest --outdir $O --cell g_tw_fc --steps 5 2>&1 | grep -v "robosuite WARNING" | grep -E "SELFTEST|Traceback|Error|error" 
K=$(python3 -c "import json;print(':'.join(map(str,json.load(open('$Z/out/ovla_rerun/keys.json'))['smoke_src_key'])))")
python -u -X faulthandler $Z/tools/ovla_qpos_rerun.py run --outdir $O --cell src --offset 0 --keys $K --part 0 --host $HOST --port $PORT \
  > $O/client_src_$SLURM_JOB_ID.log 2>&1
grep -E "^RERUN|^EP|Traceback|INFER_RETRY|^FINAL" $O/client_src_$SLURM_JOB_ID.log
python -u $Z/tools/ovla_rerun_compare.py smoke --rerun-dir $O
echo "end=$(date)"
