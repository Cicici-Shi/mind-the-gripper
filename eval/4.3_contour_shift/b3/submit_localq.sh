#!/bin/bash
# 局部响应检验正式采集的驱动（未冻结，现在会拒绝提交）。
#   bash submit_localq.sh [--dry-run] [--queries all|preclose] [--slice 15] [--procs 2]
# 切法：B1、B2 正式键（b2/keysets/formal_*.json）按 --slice 切片，每片一个数组任务，每任务同卡 --procs 个进程（片再均分）；
# 只用 A100-40（零平移逐位复现要求与第一阶段同型号），同时 2 个任务。--skip-done 可续跑。
set -eu
cd $PROJ_ROOT/bench2_simpler/b3
DRY=0; QUERIES=all; SLICE=15; PROCS=2
while [ $# -gt 0 ]; do case $1 in --dry-run) DRY=1;; --queries) QUERIES=$2; shift;; --slice) SLICE=$2; shift;; --procs) PROCS=$2; shift;; esac; shift; done
if [ $DRY = 0 ] && [ ! -f FROZEN_localq ]; then echo "FROZEN_localq 不存在，拒绝提交"; exit 1; fi
OUT=out_localq; SL=localq_slices; [ $DRY = 1 ] && { OUT=_dry_out_localq; SL=_dry_localq_slices; }
mkdir -p $SL
python3 - "$SLICE" "$SL" <<'P'
import json, sys
n, d = int(sys.argv[1]), sys.argv[2]
i = 0
for b in ("B1", "B2"):
    ks = json.load(open(f"../b2/keysets/formal_{b}.json"))
    for j in range(0, len(ks), n):
        json.dump(ks[j:j + n], open(f"{d}/{i:03d}_{b}.json", "w"), indent=1); i += 1
print(i, "slices")
P
N=$(ls $SL/*.json | wc -l)
cat > $SL/job.sbatch <<J
#!/bin/bash
#SBATCH -J b3_localq
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-$((N-1))%2
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/localq_%A_%a.log
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
nvidia-smi -L | grep -q "A100-PCIE-40GB\|A100-SXM4-40GB" || { echo "not A100-40, abort"; exit 1; }
F=\$(ls $SL/*.json | sed -n "\$((SLURM_ARRAY_TASK_ID+1))p"); B=\$(basename \$F .json | cut -d_ -f2)
python3 - \$F $PROCS <<'Q'
import json, sys
ks = json.load(open(sys.argv[1])); p = int(sys.argv[2])
for i in range(p): json.dump(ks[i::p], open(sys.argv[1][:-5] + f".p{i}.part", "w"))
Q
P=()
for i in \$(seq 0 $((PROCS-1))); do
  python localq.py --block \$B --keys \${F%.json}.p\$i.part --src ../b2/out_formal --out $OUT --queries $QUERIES --skip-done \
    > logs/localq_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\$i.log 2>&1 & P+=(\$!); sleep 8
done
waitall
J
if [ $DRY = 1 ]; then echo "dry-run：$N 个数组任务，作业脚本 $SL/job.sbatch，未提交"; exit 0; fi
squeue -h -u $USER -n b3_localq | grep -q . && { echo "已有 b3_localq 在队列，拒绝重复提交"; exit 1; }
sbatch $SL/job.sbatch
