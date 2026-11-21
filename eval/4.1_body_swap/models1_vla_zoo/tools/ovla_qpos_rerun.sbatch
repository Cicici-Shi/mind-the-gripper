#!/bin/bash
#SBATCH --job-name=ovr_cli
#SBATCH --partition=gpu
# (site-specific #SBATCH --exclude list removed)
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ovr_cli_%j.log
# OVLA re-run clients (Fable add1 4.4): the frozen GTw over-1-mm keys (out/ovla_rerun/keys.json) of cell g_tw_fc,
# shard 0, NPART client processes (keys round-robin; each process walks the whole shard so the np.random stream is
# the original one) against the NSRV=1 server of job $SRVJOB (servers/openvla_rerun_0.addr). Cancels that server
# (its own chain job) when done.
set -u
export TMPDIR=$HOME/tmp XDG_CACHE_HOME=$HOME/.cache HF_HOME=$PROJ_ROOT/vla_zoo/hf_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONFAULTHANDLER=1 HIDE_ARM=0 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset DISPLAY
Z=$PROJ_ROOT/vla_zoo; O=$Z/out/ovla_rerun/main; mkdir -p $O/logs
NPART=${NPART:-4}
export NUS_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
A=$Z/servers/openvla_rerun_0.addr; w=0
until [ -f $A ] && [ "$(awk '{print $3}' $A)" = "$SRVJOB" ]; do
  sleep 10; w=$((w + 10))
  [ $w -gt 2700 ] && { echo "server $SRVJOB never wrote $A"; exit 2; }
  squeue -h -j $SRVJOB 2>/dev/null | grep -q . || { echo "server job $SRVJOB gone"; exit 2; }
done
read -r HOST PORT SJ SGPU < $A
export NUS_SRV_HOST=$HOST NUS_SRV_GPU="$SGPU"
echo "node=$(hostname) gpu=$NUS_GPU server=$HOST:$PORT (job $SJ, $SGPU) start=$(date)"
sha256sum $Z/tools/ovla_qpos_rerun.py $Z/common/zoo_client.py $Z/common/m1_fastsync.py $Z/tools/zoo_cells.sh $PROJ_ROOT/embody_toy/embody/body_env.py $Z/out/ovla_rerun/keys.json
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oftpi; cd $HOME/tmp
PARTS=$(python3 - <<PY
import json
k = json.load(open("$Z/out/ovla_rerun/keys.json"))["keys"]
n = $NPART
print(" ".join(",".join(f"{t}:{e}" for t, e in k[i::n]) for i in range(n) if k[i::n]))
PY
)
i=0
for ks in $PARTS; do
  python -u -X faulthandler $Z/tools/ovla_qpos_rerun.py run --outdir $O --cell g_tw_fc --offset 0 --keys $ks --part $i \
    --host $HOST --port $PORT > $O/logs/client_p${i}_$SLURM_JOB_ID.log 2>&1 &
  i=$((i + 1))
done
wait
echo "clients done $(date)"
grep -hE "^RERUN|^EP|Traceback|INFER_RETRY" $O/logs/client_p*_$SLURM_JOB_ID.log
echo "QACC lines: $(cat $O/logs/client_p*_$SLURM_JOB_ID.log | grep -c QACC)"
scancel $SRVJOB; echo "server $SRVJOB cancelled; end=$(date)"
