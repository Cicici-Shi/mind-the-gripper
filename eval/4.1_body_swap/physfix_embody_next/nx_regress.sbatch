#!/bin/bash
#SBATCH --job-name=nx_regress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
# embody_next regression, all new switches off: old = embody_toy (frozen), old2 = embody_toy again (render noise floor),
# new = embody_next. Cells (vla_zoo/tools/zoo_cells.sh vars): src (plain env), sham_cg (Panda via BodyEnv),
# grpkp100 (G), g_tfc (GTfc, twin render). Suites spatial + object, tasks 0,3,7 eps 0-1, actions of the models1 OFTi
# dumps; per step qpos and md5 of both policy images (step 0 = first frame).
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
N=$PROJ_ROOT/embody_next; T=$PROJ_ROOT/embody_toy; O=$N/out/regress_$SLURM_JOB_ID; mkdir -p $O; cd $N
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date) toy_body_env=$(sha256sum $T/embody/body_env.py | cut -c1-16) next_body_env=$(sha256sum embody/body_env.py | cut -c1-16)"
CELLS=$PROJ_ROOT/vla_zoo/tools/zoo_cells.sh
R="python tools/nx_regress.py --render 1 --tasks 0,3,7 --eps 0-1 --save-img 40"
for suite in libero_spatial libero_object; do
  if [ $suite = libero_spatial ]; then RUNS=$PROJ_ROOT/vla_zoo/runs/oft_sp; P=""; else RUNS=$PROJ_ROOT/vla_zoo/runs/oft_obj; P="obj_"; fi
  for c in src sham_cg grpkp100 g_tfc; do
    set -- $(bash $CELLS vars $c); b=$1; shift; v="$*"
    for x in old old2 new; do code=$T; [ $x = new ] && code=$N
      env $v $R --suite $suite --runs $RUNS --code $code --cond $P$c --body $b --out $O/${suite}_${c}_$x > $O/${suite}_${c}_$x.log 2>&1 &
    done
  done
  wait
done
for suite in libero_spatial libero_object; do for c in src sham_cg grpkp100 g_tfc; do for x in old2 new; do
  echo "== $suite $c old vs $x"; python tools/regress_img_cmp.py $O/${suite}_${c}_old $O/${suite}_${c}_$x | tail -1; done
  python3 - $O/${suite}_${c}_old/result.jsonl $O/${suite}_${c}_new/result.jsonl <<'PY'
import json, sys
a = {(r["task"], r["ep"]): r for r in map(json.loads, open(sys.argv[1]))}
b = {(r["task"], r["ep"]): r for r in map(json.loads, open(sys.argv[2]))}
tw = [k for k in a if "twin" in a[k]]
print("   keys", len(a), len(b), "| new vs dump qpos max", max(r["qpos_max_diff"] for r in b.values()),
      "| twin series equal:", all(a[k]["twin"] == b[k]["twin"] for k in tw) if tw else "n/a",
      "| align equal:", all(a[k]["align"] == b[k]["align"] for k in a))
PY
done; done
echo "end=$(date)"
