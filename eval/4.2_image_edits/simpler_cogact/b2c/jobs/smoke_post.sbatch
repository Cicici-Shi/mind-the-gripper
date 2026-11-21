#!/bin/bash
#SBATCH -J b2c_smokepost
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/smokepost_%j.log
# 正式采集结束后的环境不变检查（第一阶段协议 §4 第 11 项）：用同一组校准键重跑定稿冒烟（四块 x 2 键 x 五格，盲成败），
# 输出到 out_smoke_post；再用 cmp_smoke.py 与定稿冒烟 out_smoke 逐条比 raw_actions、pred_chunks、env_actions、frame0_sha256。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
gpu_sampler logs/smokepost_${SLURM_JOB_ID}_gpu.csv & GS=$!
# L 从冻结清单取（Fractal 为 8，见规格「追加」；2026-11-13 前这里漏了 --lag，F 块会按 keys.py 的 4 跑而与 out_smoke 不可比）
for b in B1 B2 F1 F2; do
  L=$(python -c "import json;print(json.load(open('FROZEN_bench2_cogact'))['blocks']['$b']['L'])") || { echo "no FROZEN_bench2_cogact"; exit 1; }
  echo "== $b L=$L $(date +%T)"
  python eval_views.py --block $b --cells C0,C1,C3,C4,C5 --lag $L --keys keysets/smoke_$b.json --out out_smoke_post \
    --skip-done --blind-success 2>&1 | grep -v "$NOISE" > logs/smokepost_${SLURM_JOB_ID}_${b}.log
  grep -h "ERROR" logs/smokepost_${SLURM_JOB_ID}_${b}.log || echo "no ERROR in $b"
done
kill $GS
python cmp_smoke.py --new out_smoke_post --ref out_smoke
for b in B1 B2 F1 F2; do python gate.py --root out_smoke_post --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2_cogact --out out_smoke_post/gate_$b.json; echo; done
