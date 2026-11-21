#!/bin/bash
#SBATCH --job-name=toy_regf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# TOY1 final-code regression (protocol 9: sham, grpkp100, sham_twin, grpkp100_twin, 10 episodes each = task 0..9 ep 0;
# qpos per step and twin residual series identical; images md5 with old-vs-old repeat as noise floor).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
T=$PROJ_ROOT/embody_toy; V=$PROJ_ROOT/embody_v3; O=$T/out/regf_$SLURM_JOB_ID; mkdir -p $O; cd $T
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date) body_env_sha=$(sha256sum embody/body_env.py | cut -c1-16)"
R="python tools/regress_img.py --render 1 --eps 0-0 --save-img 300"
for spec in "sham|Panda|" "grpkp100|PandaR85|GRP_KP=100" "sham_twin|Panda|TWIN_RENDER=1" "grpkp100_twin|PandaR85|GRP_KP=100 TWIN_RENDER=1"; do
  IFS='|' read c b v <<< "$spec"
  for x in old old2 new; do code=$V; [ $x = new ] && code=$T
    env $v $R --code $code --cond $c --body $b --out $O/${c}_$x > $O/${c}_$x.log 2>&1 & done
done
wait
for c in sham grpkp100 sham_twin grpkp100_twin; do for x in old2 new; do echo "== $c old vs $x"; python tools/regress_img_cmp.py $O/${c}_old $O/${c}_$x | tail -1; done
  python3 - $O/${c}_old/result.jsonl $O/${c}_new/result.jsonl <<'PY'
import json, sys
a = {(r["task"], r["ep"]): r for r in map(json.loads, open(sys.argv[1]))}
b = {(r["task"], r["ep"]): r for r in map(json.loads, open(sys.argv[2]))}
tw = [k for k in a if "twin" in a[k]]
print("   twin pos_max/rot_max/qmiss equal:", all(a[k]["twin"]["pos_max"] == b[k]["twin"]["pos_max"] and a[k]["twin"]["rot_max"] == b[k]["twin"]["rot_max"] and a[k]["twin"]["qmiss"] == b[k]["twin"]["qmiss"] for k in tw) if tw else "n/a", "eps", len(a), len(b))
PY
done
echo "end=$(date)"
