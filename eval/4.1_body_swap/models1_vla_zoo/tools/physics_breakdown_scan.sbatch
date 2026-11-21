#!/bin/bash
#SBATCH --job-name=pb_scan_m1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pb_scan_m1_%j.log
# Read-only physics-breakdown rescan for models1 (Fable add1 section 4 item 1). CPU only, numpy only, no GPU,
# never reads success. Rescans ALL shards present of every OFTi cell (both suites) and all pi0 / OpenVLA client logs.
# Run before the 11/9 sensitivity readout, after the last OFTi object shard has finished:
#   sbatch $PROJ_ROOT/vla_zoo/tools/physics_breakdown_scan.sbatch
# Optional: PB_PUBLISHED=1 also redoes the published (third batch OFT + pi0.5) files.
Z=$PROJ_ROOT/vla_zoo
cd $Z || exit 1
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$HOME/tmp
PY=$HOME/miniforge3/bin/python3
echo "host $(hostname)  $(date)"
sha256sum tools/physics_breakdown_scan.py tools/physics_breakdown_scan.sbatch
# keep the previous outputs as dated snapshots
for f in out/physics_breakdown_m1.json out/physics_breakdown_m1_logs.json; do
  [ -f $f ] && cp -p $f ${f%.json}.prev_$(date +%m%d%H%M).json
done
set -e
$PY tools/physics_breakdown_scan.py --out out/physics_breakdown_m1.json --preset m1
$PY tools/physics_breakdown_scan.py --logs --out out/physics_breakdown_m1_logs.json
if [ "${PB_PUBLISHED:-0}" = 1 ]; then
  $PY tools/physics_breakdown_scan.py --preset published --out out/physics_breakdown_published.json
  $PY tools/physics_breakdown_scan.py --preset published --logs --out out/physics_breakdown_published_logs.json
fi
sha256sum out/physics_breakdown_m1*.json
echo "done $(date)"
