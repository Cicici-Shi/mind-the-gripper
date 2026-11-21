#!/bin/bash
# maskinfo-confirm formal submission (protocol_maskinfo_confirm.md v1 §2, §5, §7 items 5-6). Only after the freeze
# (out/FROZEN). Everything on h200, one job per shard (8 processes), order oft_spatial_c30, ofti_spatial_c30,
# oft_object_c30, ofti_object_c30, shards EP_OFFSET 30 then 35, each job afterany the previous one (all four blocks
# run whatever happens to earlier ones). SKIP_DONE=1 so a re-submission only fills holes. Optional DEP=<jobid> makes
# the first job afterany that job. Prints job ids in order.
set -e
cd $PROJ_ROOT/maskinfo_confirm
[ -e out/FROZEN ] || { echo "refusing: out/FROZEN missing (freeze first)"; exit 2; }
tail -n +2 out/FROZEN | sha256sum -c --quiet - || { echo "refusing: out/FROZEN sha check failed"; exit 2; }
[ -s out/DRIVER_EXPECT ] || { echo "refusing: out/DRIVER_EXPECT missing"; exit 2; }
export EXPECT_DRIVER=$(cat out/DRIVER_EXPECT)   # every formal job exits 7 if the node driver changed (node41 reboot)
dep=${DEP:-}; ids=()
for blk in oft:spatial ofti:spatial oft:object ofti:object; do
  IFS=: read M S <<< "$blk"
  for off in 30 35; do
    id=$(sbatch --parsable --job-name=mic_${M}_${S}_e${off} ${dep:+--dependency=afterany:$dep} \
         --export=ALL,MODEL=$M,SUITE=$S,CARD=h200,JOBS="C0:$off C1:$off C2:$off C3:$off C4:$off C5:$off C6:$off C7:$off",SKIP_DONE=1 \
         tools/mi_cells_run_c.sbatch)
    ids+=($id); dep=$id
  done
done
echo "${ids[@]}"
