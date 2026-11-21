#!/bin/bash
# Submit the OVLA re-run chain (Fable add1 4.4). Usage: bash ovla_qpos_rerun_submit.sh [READOUT_JOB=<job>]
#   ovr_wait (CPU)  --begin=2026-11-09T12:30:00 --dependency=afterany:<readout>; blocks while OVLA collection jobs exist
#   ovr_srv  (h100-47, common/run_server_multi.sbatch unchanged, MODEL=openvla_rerun NSRV=1)  afterok:ovr_wait
#   ovr_cli  (T4)   after:ovr_srv (starts once the server runs; waits for its addr file; scancels it when done)
#   ovr_diag (CPU)  afterany:ovr_cli
set -eu
Z=$PROJ_ROOT/vla_zoo; RO=${1:-<job>}
mkdir -p $Z/out/ovla_rerun/main
cd $Z
W=$(sbatch --parsable --begin=2026-11-09T12:30:00 --dependency=afterany:$RO tools/ovla_qpos_rerun_wait.sbatch)
S=$(sbatch --parsable --dependency=afterok:$W --job-name=ovr_srv --gres=gpu:h100-47:1 --time=02:30:00 \
      --output=$Z/out/ovla_rerun/main/ovr_srv_%j.log \
      --export=ALL,MODEL=openvla_rerun,NSRV=1,SERVER=zoo:zoo_openvla:$Z/openvla/adapter.py:unnorm_key=libero_spatial common/run_server_multi.sbatch)
C=$(sbatch --parsable --dependency=after:$S --export=ALL,SRVJOB=$S tools/ovla_qpos_rerun.sbatch)
D=$(sbatch --parsable --dependency=afterany:$C tools/ovla_qpos_rerun_diag.sbatch)
echo "wait $W  server $S  clients $C  diag $D  (readout $RO)" | tee -a $Z/out/ovla_rerun/main/submitted.txt
