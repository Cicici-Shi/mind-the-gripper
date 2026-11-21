#!/bin/bash
# physfix collection submit (2026-11-06): smoke gate (afterok pf_cpu) -> collection jobs afterok smoke.
# Priority order (protocol 0dd4193c 9): object main -> spatial main -> object kp20. GTloc' and bridge cells: later.
# Each job = one checkpoint x one suite, all its cells, shard-major RUNS (all cells e0 first, then e5, e10, e15).
# Usage: bash submit_pf.sh <pf_cpu jobid>
set -eu
Z=$PROJ_ROOT/vla_zoo; P=$Z/next/pf; CPU=$1
CPUDIR=$Z/out/pf/cpu_$CPU
S=$(cd $P && sbatch --parsable --dependency=afterok:$CPU --export=ALL,CPUDIR=$CPUDIR smoke_pf.sbatch)
echo "smoke $S"
runs() { local out=""; for off in 0 5 10 15; do for c in "$@"; do out="$out $c:$off"; done; done; echo ${out# }; }
sub() { # model ckpt suite hfh cells...
  local m=$1 ck=$2 su=$3 hf=$4; shift 4
  local R=$(runs "$@")
  local j=$(cd $Z/oft && sbatch --parsable --job-name=pf_${m#pf_} --dependency=afterok:$S --export=ALL,MODEL=$m,CKPT=$ck,SUITE=$su,HFH=$hf,RUNS="$R" $P/run_oft_pf.sbatch)
  echo "$j $m $su RUNS=$R"
}
OC=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
# 2026-11-06 evening: G' / GTfc' (armature) HELD: XML-level armature changes the free-air closed plateau (0.70 -> 0.415),
# decision pending. Submitted now: S, T0 (all four checkpoint x suite) + object kp20 G'' / GTfc'' (no armature).
sub pf_oftc_obj $OC libero_object $HOME/hf_cache pf_obj_sham_cg pf_obj_sham_twin pf_obj_grpkp20 pf_obj_g_tfc_kp20
sub pf_ofti_obj moojink/openvla-7b-oft-finetuned-libero-object libero_object $Z/hf_cache pf_obj_sham_cg pf_obj_sham_twin pf_obj_grpkp20 pf_obj_g_tfc_kp20
sub pf_oftc_sp $OC libero_spatial $HOME/hf_cache pf_sham_cg pf_sham_twin
sub pf_ofti_sp moojink/openvla-7b-oft-finetuned-libero-spatial libero_spatial $Z/hf_cache pf_sham_cg pf_sham_twin
