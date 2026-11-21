#!/bin/bash
# maskinfo-pi: rerun ONE incomplete formal shard whole, in a new server process (protocol 2.4 / 6: no splicing across
# server processes). Usage: tools/rerun_shard_mi_pi.sh MODEL SUITE OFFSET [dependency jobid]
# Moves the shard's outputs of ALL nine cells (npz, error npz, blind jsonl) unread to out/runs_incomplete/<stamp>/ (by
# file name only; nothing is opened) and resubmits the shard. Requires an intact out/FROZEN. Appends to SUBMITTED.txt.
set -eu
T=$PROJ_ROOT/maskinfo_pi; cd $T
M=$1; S=$2; OFF=$3; DEP=${4:-}
case "$OFF" in 20|25) ;; *) echo "OFFSET must be 20 or 25"; exit 2;; esac
[ -f out/FROZEN ] || { echo "out/FROZEN missing"; exit 2; }
sha256sum -c --quiet out/FROZEN || { echo "frozen files changed"; exit 2; }
B=${M}_${S}; STAMP=$(date +%Y%m%d_%H%M%S); INC=out/runs_incomplete/$STAMP/$B; n=0
shopt -s nullglob
for c in C0 C1 C2 C3 C4 C5 C6 C7 C8; do
  mkdir -p $INC/$c $INC/${c}_errors
  for e in $(seq $OFF $((OFF + 4))); do
    for f in out/runs/$B/$c/t*_e${e}_i*.npz; do mv "$f" $INC/$c/; n=$((n + 1)); done
    for f in out/runs_errors/$B/$c/t*_e${e}_i*.npz; do mv "$f" $INC/${c}_errors/; n=$((n + 1)); done
  done
  for f in out/runs/$B/$c/blind_e${OFF}_*.jsonl; do mv "$f" $INC/$c/; done
done
J=""; for c in C0 C1 C2 C3 C4 C5 C6 C7 C8; do J="$J $c:$OFF"; done
id=$(sbatch --parsable --nice=100 --job-name=mipi_${B}_e${OFF}_rerun ${DEP:+--dependency=afterany:$DEP} \
     --export=ALL,MODEL=$M,SUITE=$S,JOBS="${J# }",NTRIALS=5,TASKS=all tools/mi_pi_run.sbatch)
echo "$id mipi formal RERUN $B shard e$OFF (whole shard; $n files moved unread to $INC) $(date)" >> SUBMITTED.txt
echo "$id $B e$OFF rerun; moved $n files to $INC"
