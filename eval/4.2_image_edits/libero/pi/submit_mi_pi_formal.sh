#!/bin/bash
# maskinfo-pi formal submission (protocol v2 3, 5.5, 6): requires out/FROZEN and a clean sha256sum -c. Four blocks in
# the pre-registered order pi05_spatial -> pi05_object -> pi0_spatial -> pi0_object; two shards each (EP_OFFSET 20, 25,
# TRIALS 5, all ten tasks); one h200 job per shard = one server + nine clients (C0..C8); no SKIP_DONE (an incomplete
# shard is rerun whole with tools/rerun_shard_mi_pi.sh, so all nine cells of a key share one server process); each job
# afterany the previous; --nice=100 (the confirmation set has priority on the shared h200 quota). Job ids are appended
# to SUBMITTED.txt. Prints nothing about outcomes.
set -eu
T=$PROJ_ROOT/maskinfo_pi; cd $T
[ -f out/FROZEN ] || { echo "out/FROZEN missing: not frozen"; exit 2; }
sha256sum -c --quiet out/FROZEN || { echo "frozen files changed"; exit 2; }
dep=""
for blk in pi05:spatial pi05:object pi0:spatial pi0:object; do
  IFS=: read M S <<< "$blk"
  for off in 20 25; do
    J=""; for c in C0 C1 C2 C3 C4 C5 C6 C7 C8; do J="$J $c:$off"; done
    id=$(sbatch --parsable --nice=100 --job-name=mipi_${M}_${S}_e$off ${dep:+--dependency=afterany:$dep} \
         --export=ALL,MODEL=$M,SUITE=$S,JOBS="${J# }",NTRIALS=5,TASKS=all tools/mi_pi_run.sbatch)
    echo "$id mipi formal ${M}_${S} shard e$off (frozen $(sha256sum out/FROZEN | cut -c1-16)) $(date)" >> SUBMITTED.txt
    echo "$id ${M}_${S} e$off"; dep=$id
  done
done
