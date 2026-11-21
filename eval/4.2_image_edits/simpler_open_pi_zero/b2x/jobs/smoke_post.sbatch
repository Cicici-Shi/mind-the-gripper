#!/bin/bash
#SBATCH -J b2x_smokepost
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2x/logs/smokepost_%j.log
# 补充协议采集结束后的环境不变检查（b2x）：用同一组校准键重跑冒烟（B2 四格、F1 八格，盲成败，参数全取自 FROZEN_bench2_ext），
# 输出到 out_smoke_ext_post；cmp_smoke.py 与冻结前的定稿冒烟 out_smoke_ext2（作业 <job>）逐条比 raw_actions、env_actions、frame0_sha256。
source $PROJ_ROOT/bench2_simpler/b2x/jobs/common.sh
J=${SLURM_JOB_ID}
[ -f FROZEN_bench2_ext ] || { echo "FROZEN_bench2_ext 不存在"; exit 1; }
cfg() { python -c "import json,sys; m=json.load(open('FROZEN_bench2_ext'))['blocks']['$1']; print(eval(sys.argv[1]))" "$2"; }
run_cells() { b=$1; mid=$2; shift 2; P=()
  L=$(cfg $b "m['L']"); FW=$(cfg $b "','.join(map(str,m['fill_rgb']['w']))"); FB=$(cfg $b "','.join(map(str,m['fill_rgb']['L']))")
  GEO=$(cfg $b "(str(m['cell_lag']['C5'])+','+str(m['cell_lag']['C5L'])) if 'C5L' in m['cell_lag'] else ''"); LL=$(cfg $b "m['cell_lag'].get('C5L','')")
  for c in "$@"; do
    extra=""
    [ $c = C1w ] && extra="--fill-rgb $FW"
    [ $c = C1L ] && extra="--fill-rgb $FB"
    [ $c = C5L ] && extra="--lag-long $LL"
    case $c in C1|C1w|C1L) extra="$extra --fill-stats" ;; esac
    [ -n "$GEO" ] && extra="$extra --geom-lags $GEO"
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke_ext_post --lag $L --save-frames 0 \
      --blind-success --skip-done $extra > logs/smokepost_${J}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 10
  done; waitall
  grep -h "ERROR" logs/smokepost_${J}_${b}_*.log || echo "no ERROR in $b $*"
}
run_cells B2 7 C0 C1 C1w C1L
run_cells F1 20 C0 C1 C1w C1L
run_cells F1 20 C3 C4 C5 C5L
python cmp_smoke.py --new out_smoke_ext_post --ref out_smoke_ext2
for b in B2 F1; do python gate.py --root out_smoke_ext_post --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2_ext --ref-root ../b2/out_smoke_v11 --out out_smoke_ext_post/gate_$b.json; echo; done
