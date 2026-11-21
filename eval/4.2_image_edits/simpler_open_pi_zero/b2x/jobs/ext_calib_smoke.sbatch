#!/bin/bash
#SBATCH -J b2x_calsmoke
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2x/logs/calsmoke_%j.log
# 补充协议 v1 冻结前（b2x）：
#  1 校准：B2、F1 各 10 个校准键重跑原图格（C0），记整回合 G 与 G 外一圈的颜色计数（FL、操纵强度、环反差），
#    F1 另记 4、8、12 步按实际显示轮廓的几何量；盲成败。ext_calib.py 汇总，并与第一阶段校准记录逐位比对；
#  2 生成冻结清单候选（freeze_manifest.py；更长滞后按规则选）；
#  3 冒烟：B2、F1 各 2 个校准键 x 本扩展全部格（B2 四格，F1 八格），参数全取自候选清单，盲成败；
#    门槛全项（候选清单），重跑的旧格与第一阶段定稿冒烟 b2/out_smoke_v11 逐位比对。
source $PROJ_ROOT/bench2_simpler/b2x/jobs/common.sh
mkdir -p logs
J=${SLURM_JOB_ID}
gpu_sampler logs/calsmoke_${J}_gpu.csv & GS=$!
echo "== 1 校准键原图格"
P=()
python eval_views.py --block B2 --cell C0 --keys keysets/calib10_B2.json --out out_calib_ext --lag 8 --color-stats \
  --blind-success --skip-done > logs/calsmoke_${J}_cal_B2.log 2>&1 &
P+=($!); sleep 10
python eval_views.py --block F1 --cell C0 --keys keysets/calib10_F1.json --out out_calib_ext --lag 4 --color-stats \
  --geom-lags 4,8,12 --blind-success --skip-done > logs/calsmoke_${J}_cal_F1.log 2>&1 &
P+=($!); waitall
grep -h "ERROR" logs/calsmoke_${J}_cal_*.log || echo "no ERROR in calib"
python -W ignore ext_calib.py --root out_calib_ext --ref-root ../b2/out_calib --out calib_ext.json > logs/calsmoke_${J}_ext_calib.txt
cat logs/calsmoke_${J}_ext_calib.txt
echo "== 2 候选清单"
python freeze_manifest.py > logs/calsmoke_${J}_manifest.txt && echo "candidate written"
cfg() { python -c "import json,sys; m=json.load(open('FROZEN_bench2_ext.candidate'))['blocks']['$1']; print(eval(sys.argv[1]))" "$2"; }
echo "== 3 冒烟"
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
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke_ext --lag $L --save-frames 0,$mid \
      --blind-success --skip-done $extra > logs/calsmoke_${J}_smoke_${b}_${c}.log 2>&1 &
    P+=($!); sleep 10
  done; waitall
  grep -h "ERROR" logs/calsmoke_${J}_smoke_${b}_*.log || echo "no ERROR in $b $*"
}
run_cells B2 7 C0 C1 C1w C1L
run_cells F1 20 C0 C1 C1w C1L
run_cells F1 20 C3 C4 C5 C5L
kill $GS
grep -il "llvmpipe\|lavapipe" logs/calsmoke_${J}_* && echo "SOFTWARE RENDERER SEEN" || echo "no software renderer in logs"
for b in B2 F1; do python gate.py --root out_smoke_ext --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2_ext.candidate --ref-root ../b2/out_smoke_v11 --out out_smoke_ext/gate_$b.json; echo; done
