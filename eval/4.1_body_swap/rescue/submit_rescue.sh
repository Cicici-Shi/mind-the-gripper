#!/bin/bash
# rescue pilot (protocol_rescue_pilot.md 78c1e1d3..., frozen 2026-11-08), execution sub-agent.
# Reuses the physfix collection sbatch and launcher UNCHANGED:
#   $PROJ_ROOT/vla_zoo/next/pf/run_oft_pf.sbatch 12557635...  (a100-40 whole card, NPAR=2, EMBODY_DIR=embody_next,
#   M1_FASTSYNC=1, SKIP_DONE=1, blind launcher next/pf/launch_pf.py 7afeb3c3...), cell switches next/cells_next.sh.
# Cells (protocol 2): S = pf_sham_cg, G2 = pf_grpkp20, GTfc2 = pf_g_tfc_kp20 (switches identical in cells_next.sh
# e38bce0a and a93b269c). Outputs go to NEW model dirs runs/rs_{oftc,ofti}_sp/<cell>/ so pf_*_sp is never written.
# Usage: bash submit_rescue.sh smoke             -> 2 smoke jobs (MAX_TASKS=1 TRIALS=2: keys t0 e0,e1 per cell)
#        bash submit_rescue.sh full <afterok-dep>  -> 2 collection jobs (200 keys per cell, shard-major e0 e5 e10 e15)
set -eu
Z=$PROJ_ROOT/vla_zoo; PF=$Z/next/pf; R=$PROJ_ROOT/rescue
CELLS="pf_sham_cg pf_grpkp20 pf_g_tfc_kp20"
OC=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
OI=moojink/openvla-7b-oft-finetuned-libero-spatial
want() { [ "$(sha256sum $1 | cut -c1-16)" = "$2" ] || { echo "sha mismatch $1"; exit 4; }; }
want $PF/run_oft_pf.sbatch 12557635173c9403; want $PF/launch_pf.py 7afeb3c317ec38d6
want $Z/next/cells_next.sh e38bce0ab4b057be; want $Z/oft/run_libero_eval_m1.py a65fc6d08c481471
want $PROJ_ROOT/embody_next/embody/body_env.py 38a1b81a3661312f; want $PROJ_ROOT/embody_next/embody/robots.py 28e6722e57531ba5
want $R/protocol_rescue_pilot.md 78c1e1d3a20166ca
mode=$1; dep=${2:-}
sub() { # model ckpt hfh runs trials extra-export
  local m=$1 ck=$2 hf=$3 runs=$4 tr=$5 ex=$6
  case $m in pf_*) echo "refuse: $m would share a physfix dir"; exit 5;; esac
  [ -e $Z/runs/$m ] && { echo "refuse: $Z/runs/$m exists"; exit 5; }
  local d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  local j=$(cd $Z/oft && sbatch --parsable --job-name=rs_${m#rs_} $d \
     --export=ALL,MODEL=$m,CKPT=$ck,SUITE=libero_spatial,HFH=$hf,TRIALS=$tr,RUNS="$runs"$ex $PF/run_oft_pf.sbatch)
  echo "$j $m RUNS=$runs TRIALS=$tr$ex"
}
case $mode in
  smoke) RS=""; for c in $CELLS; do RS="$RS $c:0"; done; RS=${RS# }
         sub rs_smoke_oftc_sp $OC $HOME/hf_cache "$RS" 2 ",MAX_TASKS=1"
         sub rs_smoke_ofti_sp $OI $Z/hf_cache "$RS" 2 ",MAX_TASKS=1" ;;
  full)  RS=""; for off in 0 5 10 15; do for c in $CELLS; do RS="$RS $c:$off"; done; done; RS=${RS# }
         sub rs_oftc_sp $OC $HOME/hf_cache "$RS" 5 ""
         sub rs_ofti_sp $OI $Z/hf_cache "$RS" 5 "" ;;
  *) echo "usage"; exit 2;;
esac
