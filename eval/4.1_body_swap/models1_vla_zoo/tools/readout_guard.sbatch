#!/bin/bash
#SBATCH --job-name=m1_guard
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.log
# Guard for the one-shot models1 readout (2026-11-05, coordinator): if the driver has not reported every shard complete,
# or any m1 collection job is still queued/running, HOLD the readout <job> (and its sens child <job>) so a partial
# data set is never read. Release by hand with `scontrol release <job>` once complete. Reads only the driver log / squeue.
Z=$PROJ_ROOT/vla_zoo
last=$(grep -E "tick.*shards complete" $Z/out/m1_driver.log | tail -1)
done_n=$(echo "$last" | sed -E "s/.* ([0-9]+)\/([0-9]+) shards complete.*/\1/"); tot=$(echo "$last" | sed -E "s/.* ([0-9]+)\/([0-9]+) shards complete.*/\2/")
busy=$(squeue -u $USER -h -o "%j" | grep -cE "^m1_(oft|d_|srv|pi0|ovla)")
echo "$(date) last tick: $last | busy collection jobs: $busy"
if [ -n "$GUARD_LOGONLY" ]; then echo "GUARD LOG-ONLY: $done_n/$tot, busy $busy"; echo "$(date +%m-%d\ %H:%M) GUARD check (log only): $done_n/$tot shards, busy $busy" >> $Z/out/m1_driver.log
elif [ "$done_n" = "$tot" ] && [ -n "$tot" ] && [ "$busy" = "0" ]; then
  echo "GUARD OK: $done_n/$tot complete, readout left as scheduled"
else
  scontrol hold <job> && echo "GUARD HOLD: <job> held ($done_n/$tot, busy $busy); release with scontrol release <job>"
  echo "$(date +%m-%d\ %H:%M) GUARD HOLD readout <job>: $done_n/$tot shards, busy $busy" >> $Z/out/m1_driver.log
fi
