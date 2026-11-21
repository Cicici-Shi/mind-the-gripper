#!/bin/bash
#SBATCH -J b2_tput
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/tput_%j.log
# 吞吐：同一张卡上并行 1、2、4 个进程（C4，每步多一次渲染，是最重的格），每个进程 2 个校准键，记每回合耗时与显存。
# 成败不写（--blind-success）。B1（120 步）与 F1（80 步）各测一轮。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
mkdir -p keysets
python - <<'PY'
import json, sys
sys.path.insert(0, ".")
import keys as K
for b in ("B1", "F1"):
    ks = K.make_keys(b, "calib")
    if b == "F1":
        ks = K.spread(ks, 8)
    for i in range(4):
        json.dump(ks[2 * i: 2 * i + 2], open(f"keysets/tput_{b}_{i}.json", "w"))
PY
for b in B1 F1; do for n in 1 2 4; do
  echo "== $b N=$n"; t0=$(date +%s)
  gpu_sampler logs/tput_${SLURM_JOB_ID}_${b}_N${n}_gpu.csv & GS=$!
  P=()
  for i in $(seq 0 $((n-1))); do
    python eval_views.py --block $b --cell C4 --keys keysets/tput_${b}_$i.json --out out_tput/N$n --blind-success \
      > logs/tput_${SLURM_JOB_ID}_${b}_N${n}_$i.log 2>&1 &
    P+=($!)
  done; waitall; kill $GS
  echo "wall $(( $(date +%s) - t0 ))s"; grep -h "model loaded\|done \|peak_vram\|ERROR" logs/tput_${SLURM_JOB_ID}_${b}_N${n}_*.log
  sort -t, -k1 -n logs/tput_${SLURM_JOB_ID}_${b}_N${n}_gpu.csv | tail -1
done; done
