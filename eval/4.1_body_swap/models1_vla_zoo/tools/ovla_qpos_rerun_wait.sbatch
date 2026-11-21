#!/bin/bash
#SBATCH --job-name=ovr_wait
#SBATCH -p normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ovr_wait_%j.log
# Chain head of the OVLA re-run (Fable add1 4.4). Submitted with --begin=2026-11-09T12:30 --dependency=afterany:<readout>.
# Blocks until the OVLA full collection is over: no job of this user named m1_d_openvla / m1_srvd_openvla /
# m1_srvN_ovla / m1_driver is queued or running (the driver may submit new OVLA clients until it stops at 11/9 12:00,
# so a fixed afterok on today's job ids would not cover them). Needs two consecutive empty polls 2 min apart, so a
# driver tick that is between scancel/sbatch is not mistaken for the end. Exit 0 lets the GPU jobs start (afterok).
set -u
PAT='^(m1_d_openvla|m1_srvd_openvla|m1_srvN_ovla|m1_driver)$'
empty=0
while [ $empty -lt 2 ]; do
  n=$(squeue -h -u $USER -o %j | grep -cE "$PAT")
  if [ "$n" = 0 ]; then empty=$((empty + 1)); else empty=0; echo "$(date +%F_%T) OVLA collection jobs still queued: $n"; fi
  [ $empty -lt 2 ] && sleep 120
done
echo "$(date +%F_%T) OVLA collection over; releasing the re-run"
