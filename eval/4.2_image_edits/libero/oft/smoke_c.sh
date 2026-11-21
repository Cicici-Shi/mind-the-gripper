#!/bin/bash
# maskinfo-confirm entry smoke (protocol_maskinfo_confirm.md §5, §6, §7 item 6): 2 models x 2 suites x 8 cells x 1
# episode on ALREADY-READ keys only (Spatial task 0 ep 20, Object task 0 ep 0), h200, output out/smoke_runs/ (never
# mixed with formal runs). Only fields / interface are inspected (prereg --check-fields-only, gate label fields);
# outcomes are never printed. Optional DEP=<jobid>: first job afterany it. Jobs chained afterany (one h200 at a time).
set -e
cd $PROJ_ROOT/maskinfo_confirm
dep=${DEP:-}
for blk in oft:spatial:20 ofti:spatial:20 oft:object:0 ofti:object:0; do
  IFS=: read M S off <<< "$blk"
  g="C0:$off C1:$off C2:$off C3:$off C4:$off C5:$off C6:$off C7:$off"
  id=$(sbatch --parsable --job-name=mic_smoke_${M}_${S} --time=00:59:00 ${dep:+--dependency=afterany:$dep} \
       --export=ALL,MODEL=$M,SUITE=$S,CARD=h200,JOBS="$g",NTRIALS=1,MAX_TASKS=1,OUTROOT=$PROJ_ROOT/maskinfo_confirm/out/smoke_runs tools/mi_cells_run_c.sbatch)
  echo $id; dep=$id
done
