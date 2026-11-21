#!/bin/bash
# TOY1 formal closed-loop submission (protocol_toy1.md section 3 / 10 + decisions): first shards (ep 40-44) of all 7
# cells, then the second shards (45-49) with --dependency=afterany on the first-shard job of the same card group.
# Groups (one GPU each, processes share the card): A = toy_s toy_g toy_g_tfc toy_g_hid ; B = toy_g_t toy_s_hid toy_g_cam
# usage: GA=a100-80 GB=h100-96 bash tools/submit_toy_formal.sh   (GPU types chosen from what is free at submission)
set -e
cd $PROJ_ROOT/embody_toy
GA=${GA:-a100-80}; GB=${GB:-h200-141}
ex() { echo "node39,node38"; }  # H100 NVL (node29) EGL read_pixels aborts (smoke <job>, node37): never used
A="toy_s toy_g toy_g_tfc toy_g_hid"; B="toy_g_t toy_s_hid toy_g_cam"
jobs_of() { local o=$1; shift; for c in "$@"; do printf "%s:%s " $c $o; done; }
a1=$(sbatch --parsable --job-name=toyc_A40 --gres=gpu:$GA:1 --exclude=$(ex $GA) --export=ALL,JOBS="$(jobs_of 40 $A)" tools/toy_cells_run.sbatch)
b1=$(sbatch --parsable --job-name=toyc_B40 --gres=gpu:$GB:1 --exclude=$(ex $GB) --export=ALL,JOBS="$(jobs_of 40 $B)" tools/toy_cells_run.sbatch)
a2=$(sbatch --parsable --job-name=toyc_A45 --dependency=afterany:$a1 --gres=gpu:$GA:1 --exclude=$(ex $GA) --export=ALL,JOBS="$(jobs_of 45 $A)" tools/toy_cells_run.sbatch)
b2=$(sbatch --parsable --job-name=toyc_B45 --dependency=afterany:$b1 --gres=gpu:$GB:1 --exclude=$(ex $GB) --export=ALL,JOBS="$(jobs_of 45 $B)" tools/toy_cells_run.sbatch)
echo "A40=$a1 ($GA) B40=$b1 ($GB) A45=$a2 B45=$b2"
