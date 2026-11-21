#!/bin/bash
#SBATCH -J b2x_smoke11
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:35:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2x/logs/smoke11_%j.log
# 补充协议 v1.1 冻结前（b2x）：中灰换白色后，用定稿代码重跑冒烟（B2、F1 各 2 个校准键 x 本扩展全部格，盲成败），
# 参数全取自候选清单；门槛全项（候选清单），重跑的旧格与第一阶段定稿冒烟 b2/out_smoke_v11 逐位比对。
source $PROJ_ROOT/bench2_simpler/b2x/jobs/common.sh
J=${SLURM_JOB_ID}
cfg() { python -c "import json,sys; m=json.load(open('FROZEN_bench2_ext.candidate'))['blocks']['$1']; print(eval(sys.argv[1]))" "$2"; }
run_cells() { b=$1; mid=$2; shift 2; P=()
  L=$(cfg $b "m['L']"); FW=$(cfg $b "','.join(map(str,m['fill_rgb']['w']))"); FL=$(cfg $b "','.join(map(str,m['fill_rgb']['L']))")
  GEO=$(cfg $b "(str(m['cell_lag']['C5'])+','+str(m['cell_lag']['C5L'])) if 'C5L' in m['cell_lag'] else ''")
  LL=$(cfg $b "m['cell_lag'].get('C5L','')")
  echo "$b L=$L FW=$FW FL=$FL GEO=$GEO LL=$LL"
  for c in "$@"; do
    extra=""
    [ $c = C1w ] && extra="--fill-rgb $FW"
    [ $c = C1L ] && extra="--fill-rgb $FL"
    [ $c = C5L ] && extra="--lag-long $LL"
    case $c in C1|C1w|C1L) extra="$extra --fill-stats" ;; esac
    [ -n "$GEO" ] && extra="$extra --geom-lags $GEO"
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke_ext2 --lag $L --save-frames 0,$mid \
      --blind-success --skip-done $extra > logs/smoke11_${J}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 10
  done; waitall
  grep -h "ERROR" logs/smoke11_${J}_${b}_*.log || echo "no ERROR in $b $*"
}
run_cells B2 7 C0 C1 C1w C1L
run_cells F1 20 C0 C1 C1w C1L
run_cells F1 20 C3 C4 C5 C5L
grep -il "llvmpipe\|lavapipe" logs/smoke11_${J}_* && echo "SOFTWARE RENDERER SEEN" || echo "no software renderer in logs"
for b in B2 F1; do python gate.py --root out_smoke_ext2 --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2_ext.candidate --ref-root ../b2/out_smoke_v11 --out out_smoke_ext2/gate_$b.json; echo; done
python cmp_smoke.py --new out_smoke_ext2 --ref out_smoke_ext 2>&1 | tail -4
