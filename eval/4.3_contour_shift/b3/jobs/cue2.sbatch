#!/bin/bash
#SBATCH -J b3_cue2
#SBATCH -p gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/cue2_%j.log
# b3 按草案 protocol_cue_draft_v0 与专家意见：参考轨迹回放两遍（b2/out_calib 的 10+10 个校准键 C0，只读），
# 卡：Titan RTX（gpu:nv，24 GB，策略进程同卡最多 3 个）；h100-47 是 MIG 切片，被 common.sh 的整卡检查拒绝（作业 <job>）。
# 算 q_open、位移分布、模板/偏移/重排统计与状态逐位核对；再在 2 个校准键上冒烟新格（策略，盲成败）并出对照图。
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
gpu_sampler logs/cue2_${SLURM_JOB_ID}_gpu.csv & GS=$!
R=../b2/out_calib
python - <<'P'
import json
f = json.load(open("../b2/keysets/calib10_F1.json"))
for i in range(2): json.dump(f[i::2], open(f"keysets/ref_F1_part{i}.json", "w"), indent=1)
b = json.load(open("../b2/keysets/calib10_B2.json"))
for i in range(2): json.dump(b[i::2], open(f"keysets/ref_B2_part{i}.json", "w"), indent=1)
P
# 第二次提交：第 1 遍已由作业 <job> 完成（cue_params.json 已生成）；那次第 2 遍用改关节的模板路径，复原不逐位，
# 结果移到 out_replay/*/pass2_qpos 留作证据。这里只重跑第 2 遍（模板改为不碰物理的 renderpose 路径）。
for pas in 2; do
  P=()
  EXTRA=""; [ $pas = 2 ] && EXTRA="--params cue_params.json"
  for i in 0 1; do
    python replay.py --block B2 --keys keysets/ref_B2_part$i.json --ref $R --out out_replay --pass $pas --lags 8,16 $EXTRA > logs/cue2_${SLURM_JOB_ID}_p${pas}_B2_$i.log 2>&1 & P+=($!)
    python replay.py --block F1 --keys keysets/ref_F1_part$i.json --ref $R --out out_replay --pass $pas --lags 4,8,12 $EXTRA > logs/cue2_${SLURM_JOB_ID}_p${pas}_F1_$i.log 2>&1 & P+=($!)
  done
  waitall
  grep -h "Traceback\|Error" logs/cue2_${SLURM_JOB_ID}_p${pas}_*.log | head; grep -h "\[replay\]" logs/cue2_${SLURM_JOB_ID}_p${pas}_*.log
  if [ $pas = 1 ]; then python cue_calib.py params --root out_replay --blocks B2:8,16 F1:4,8,12 --out cue_params.json --table cue_disp.json; fi
done
python cue_calib.py report --root out_replay --blocks B2 F1 --params cue_params.json --out cue_report.json
# ---- 冒烟（策略，盲成败）
for b in B2 F1; do
  if [ $b = B2 ]; then LL=8; else LL=4; fi
  python cue_smoke_specs.py cue_params.json $b $LL sheet > sheet_$b.json
  P=(); n=0
  python eval_views.py --block $b --cell C0 --tag C0_sheet --keys keysets/smoke_$b.json --out out_smoke3 --blind-success --skip-done \
    --sheet-queries 2,5,8,12,20 --sheet-specs sheet_$b.json > logs/cue2_${SLURM_JOB_ID}_smk_${b}_C0.log 2>&1 & P+=($!); n=1
  while IFS='|' read -r t c v; do
    python eval_views.py --block $b --cell $c --var "$v" --tag $t --keys keysets/smoke_$b.json --out out_smoke3 --blind-success --skip-done \
      > logs/cue2_${SLURM_JOB_ID}_smk_${b}_$t.log 2>&1 & P+=($!); n=$((n+1)); sleep 5
    if [ $n -ge 3 ]; then waitall; P=(); n=0; fi
  done < <(python cue_smoke_specs.py cue_params.json $b $LL cells)
  waitall
done
grep -l "Traceback" logs/cue2_${SLURM_JOB_ID}_smk_*.log || echo "no Traceback in smoke"
kill $GS
