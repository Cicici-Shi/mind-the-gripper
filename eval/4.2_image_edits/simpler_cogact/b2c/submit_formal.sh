#!/bin/bash
# bench2 第二阶段（CogACT）正式采集驱动。由第一阶段 b2/submit_formal.sh 改写。可断点续跑：每次运行只为缺记录的（块，格，键）出作业。
#
#   bash submit_formal.sh             真提交：要求 FROZEN_bench2_cogact 存在，且当前代码与键清单的哈希与它一致
#   bash submit_formal.sh --dry-run   空跑：不要求冻结标记，只打印计划，把切片和作业脚本生成到一个临时目录，不提交
#
# 卡型（协调方 2026-11-13）：多种卡型一起用，同一个键的五格必须在同一型号上跑。键 -> 型号由冻结清单 gpu_assignment 给定
#   （gpu_groups.GPU_PATTERN 按键序号定），本脚本只按它把缺的（格，键）派到该型号的 GPU 池：
#     A100-PCIE-40GB：gres a100-40，每卡 1 个进程，同时至多 A100_CONC 个作业（默认 1：协调方 2026-11-13，WidowX 主实验同时在用 A100-40）；
#     H100 NVL：协调方 2026-11-13 傍晚定只用 node32 节点（gres h100-47，每卡 1 个进程，同时至多 H47_CONC 个作业，默认 2）；
#       h100-96 池不再用（node37 渲染卡死，其余 h100-96 节点未获准）。
#       两种 gres 是同一型号，同一型号内的切片按进程数比例分配。
#   CogACT fp32 约 31 GB，载入一次约 3 到 4 分钟，所以一个进程跑同一块的多个（格，键）（一份切片）。
# 坏节点：作业开头 common.sh 查 MIG、型号，再做渲染自检（90 秒内出帧且第 0 帧哈希等于已知值），不过就往
#   formal_state/node_fail.tsv 记一行并退出，不跑任何键；本脚本每次提交时把这里出现过的节点加进 --exclude。
#   BASE_EXCLUDE 里写死的是已知坏节点：node39、node37（渲染卡死）、node31（h100-47 开着 MIG）。
# 冻结清单（json）：真提交用 FROZEN_bench2_cogact（协调方由 .candidate 核对后改名，本脚本不创建它）；空跑在它不存在时用 .candidate。
# 键清单：keysets/formal_<块>.json（哈希记在清单里，与第一阶段相同）；真提交首次运行时复制成 formal_state/keys_<块>.json。
# eval_views.py 带 --skip-done，记录先写临时文件再改名，所以重复提交或中途被杀都不会出半截记录。日志里不打印成败。
set -euo pipefail
B2C=$PROJ_ROOT/bench2_cogact/b2c
FROZEN=$B2C/FROZEN_bench2_cogact
CAND=$B2C/FROZEN_bench2_cogact.candidate
OUT=$B2C/out_formal
RUN_MIN=${RUN_MIN:-48}      # 每个进程留给回合的分钟数（作业 60 分钟，扣载入约 4 分钟、渲染自检与余量）
CHUNK=${CHUNK:-}            # 每份切片的（格，键）数上限；不设则按每回合秒数算；补采卡住时可设 1
H47_CONC=${H47_CONC:-2}     # h100-47（只在 node32）同时作业数，默认 2；GPU_PATTERN 按 A100 一张、h100-47 两张定
H100_NODES=node32          # 协调方 2026-11-13 傍晚：H100 只用 node32
A100_CONC=${A100_CONC:-1}   # a100-40 同时作业数（默认 1，协调方放开后可设 2；GPU_PATTERN 按 1 张定）
BASE_EXCLUDE_A100="node39,node38,node28"   # a100-40 的 gres 在 node23 上是 MIG 切片
BASE_EXCLUDE_H100="node39,node37,node31"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

cd "$B2C"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_cogact
export PYTHONDONTWRITEBYTECODE=1

if [ $DRY = 0 ]; then
  if [ ! -f "$FROZEN" ]; then
    echo "拒绝提交：$FROZEN 不存在（协议未冻结）。" >&2; exit 1
  fi
  MANIFEST=$FROZEN
  ST=$B2C/formal_state
  if ! QJOBS=$(squeue -u "$USER" -h -o "%j"); then
    echo "拒绝提交：squeue 失败，无法确认没有在跑的 b2cf_ 作业。" >&2; exit 1
  fi
  if [ -n "$(printf '%s\n' "$QJOBS" | awk '/^b2cf_/ {print; exit}')" ]; then
    echo "已有 b2cf_ 作业在队列里，先等它们结束（避免同一键被两个作业同时跑）。" >&2; exit 1
  fi
else
  if [ -f "$FROZEN" ]; then MANIFEST=$FROZEN; else MANIFEST=$CAND; fi
  [ -f "$MANIFEST" ] || { echo "空跑也需要冻结清单或候选：$FROZEN / $CAND 都不存在（先跑 freeze_manifest.py）。" >&2; exit 1; }
  ST=$(mktemp -d "$B2C/_dryrun_XXXXXX")
  echo "--dry-run：清单 $MANIFEST，临时目录 $ST（不提交）"
fi

python - "$MANIFEST" "$B2C" "$DRY" <<'PY'
import hashlib, json, os, sys
man, b2c, dry = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3] == "1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f"代码 {f}" for f, h in man["code_sha256"].items() if sha(os.path.join(b2c, f)) != h]
bad += [f"键清单 {b}" for b, h in man["keylist_sha256"].items() if sha(os.path.join(b2c, "keysets", f"formal_{b}.json")) != h]
if bad:
    print(("空跑提示" if dry else "拒绝提交") + "：与冻结清单不一致：" + "，".join(bad), file=sys.stderr)
    sys.exit(0 if dry else 1)
print("代码与键清单的哈希与冻结清单一致", file=sys.stderr)
PY

mkdir -p "$ST/chunks"
touch "$ST/node_fail.tsv"
[ $DRY = 1 ] || mkdir -p "$B2C/logs/formal"
for b in B1 B2 F1 F2; do
  if [ ! -f "$ST/keys_$b.json" ]; then
    [ -f "$B2C/keysets/formal_$b.json" ] || { echo "拒绝提交：缺 $B2C/keysets/formal_$b.json（冻结的键清单）" >&2; exit 1; }
    cp "$B2C/keysets/formal_$b.json" "$ST/keys_$b.json"
  fi
  cmp -s "$ST/keys_$b.json" "$B2C/keysets/formal_$b.json" || { echo "拒绝提交：$ST/keys_$b.json 与冻结的键清单不一致" >&2; exit 1; }
done
BAD_NODES=$(cut -f2 "$ST/node_fail.tsv" | sort -u | paste -sd, -)

TAG=$(date +%Y%m%d_%H%M%S)
# 每个 GPU 池一份清单 manifest_<TAG>_<池>.txt：每行一个数组任务，空格分隔的「块:L:线宽:切片文件」，至多 每卡进程数 个
python - "$ST" "$OUT" "$TAG" "$RUN_MIN" "${CHUNK:-0}" "$MANIFEST" <<'PY'
import json, os, sys
import numpy as np
st, out, tag, run_min, chunk, frozen = sys.argv[1:7]
run_min, chunk = int(run_min), int(chunk)
man = json.load(open(frozen))
blocks, assign = man["blocks"], man["gpu_assignment"]
A100, H100 = "NVIDIA A100-PCIE-40GB", "NVIDIA H100 NVL"
assert set(man["gpu_models"]) == {A100, H100}, man["gpu_models"]
# 每回合秒数：实测每格平均值乘约 1.15 的余量（作业 <job>、<job>、<job>、<job> 在 A100-40 上一卡一进程；
#   H100 NVL（gres h100-47，只在 node32）一卡一进程：作业 <job> 校准 C0 每回合 B 约 28 秒、F 约 14.7 秒；作业 <job> 定稿冒烟
#   B 每键五格约 174 秒、F 约 108 秒，但有单回合 115 到 121 秒的离群（h100-47 是半张卡，同卡有别人的作业）。切片按 A100 的秒数定（保守））
SEC = {A100: {"B": {"C0": 34, "C1": 34, "C3": 41, "C4": 43, "C5": 44}, "F": {"C0": 26, "C1": 26, "C3": 30, "C4": 32, "C5": 32}},
       H100: {"B": {"C0": 34, "C1": 34, "C3": 41, "C4": 43, "C5": 44}, "F": {"C0": 26, "C1": 26, "C3": 30, "C4": 32, "C5": 32}}}
# (gres, 每卡进程数, 份额)
POOLS = {A100: [("a100-40", 1, 1.0)], H100: [("h100-47", 1, 1.0)]}
lines = {g: [] for pools in POOLS.values() for g, _, _ in pools}
summary = []
for model in (A100, H100):
    chunks = []
    for b in ("B1", "B2", "F1", "F2"):
        keys = [k for k in json.load(open(os.path.join(st, f"keys_{b}.json"))) if assign[b][k["key"]] == model]
        L, lw = int(blocks[b]["L"]), int(blocks[b]["line_width"])
        sec = SEC[model][b[0]]
        miss = []
        for k in keys:  # 键在外层：同一键的五格尽量在同一份切片里（同型号即可，不要求同一份）
            for c in ("C0", "C1", "C3", "C4", "C5"):
                p = os.path.join(out, b, c, k["key"] + ".npz")
                ok = False
                if os.path.exists(p):
                    try:
                        with np.load(p) as z:
                            ok = bool(z["complete"])
                    except Exception:
                        ok = False
                if not ok:
                    miss.append({"cell": c, "key": k})
        cur, t = [], 0
        for m in miss:
            s = sec[m["cell"]]
            if cur and ((chunk and len(cur) >= chunk) or (not chunk and t + s > run_min * 60)):
                chunks.append((b, L, lw, cur)); cur, t = [], 0
            cur.append(m); t += s
        if cur:
            chunks.append((b, L, lw, cur))
        summary.append(f"{model} {b}: 键 {len(keys)}，缺 {len(miss)} 个（格，键）")
    # 同一型号内按份额把切片分给各池：累计份额最少的池先拿
    pools = POOLS[model]
    got = {g: [] for g, _, _ in pools}
    for ch in chunks:
        g = min(pools, key=lambda p: (len(got[p[0]]) / p[2], p[0]))[0]
        got[g].append(ch)
    for g, procs, _ in pools:
        files = []
        for i, (b, L, lw, part) in enumerate(got[g]):
            f = os.path.join(st, "chunks", f"{tag}_{g}_{i:03d}_{b}.json")
            with open(f, "w") as fh:
                json.dump(part, fh)
            files.append(f"{b}:{L}:{lw}:{f}")
        for i in range(0, len(files), procs):
            lines[g].append(" ".join(files[i:i + procs]))
for g, ls in lines.items():
    with open(os.path.join(st, f"manifest_{tag}_{g}.txt"), "w") as fh:
        for l in ls:
            fh.write(l + "\n")
    summary.append(f"池 {g}：{len(ls)} 个作业，{sum(len(l.split()) for l in ls)} 份切片")
print("\n".join(summary), file=sys.stderr)
PY

submit_pool() {  # 池名 每卡进程数 同时作业数 exclude [nodelist]
  local pool=$1 procs=$2 conc=$3 excl=$4 nodes=${5:-}
  local NL=""; [ -n "$nodes" ] && NL="#SBATCH --nodelist=$nodes"
  local MAN="$ST/manifest_${TAG}_${pool}.txt"
  local NT; NT=$(wc -l < "$MAN" | tr -d ' ')
  [ "$NT" = "0" ] && { echo "池 $pool：没有缺的。"; return 0; }
  [ -n "$BAD_NODES" ] && excl="$excl,$BAD_NODES"
  local MEM=$((procs * 50 + 50))   # 载入时一个进程要把 30 GB 检查点读进内存（峰值约 55 GB）；一卡三进程同时载入时 160G 被 OOM 杀过（作业 <job>）
  local JOB=$ST/job_${TAG}_${pool}.sbatch
  cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH -J b2cf_${pool}_$TAG
#SBATCH -p gpu
#SBATCH --gres=gpu:${pool}:1
# (site-specific #SBATCH --exclude list removed)
$NL
#SBATCH --cpus-per-task=$((procs * 5))
#SBATCH --mem=${MEM}G
#SBATCH --time=01:00:00
#SBATCH --array=0-$((NT - 1))%$conc
#SBATCH -o $B2C/logs/formal/%x_%A_%a.log
export B2C_NODE_FAIL=$ST/node_fail.tsv
source $B2C/jobs/common.sh
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
render_selfcheck
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" $MAN)
P=()
for t in \$LINE; do
  IFS=: read b L lw f <<< "\$t"
  python eval_views.py --block \$b --plan \$f --out $OUT --lag \$L --line-width \$lw --skip-done \
    > $B2C/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\$(basename \$f .json).log 2>&1 &
  P+=(\$!)
  # 错开载入：等这个进程打出 model loaded（载入完、检查点内存已释放）或退出，再起下一个，至多等 10 分钟
  for i in \$(seq 1 120); do grep -q "model loaded" $B2C/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\$(basename \$f .json).log 2>/dev/null && break; kill -0 \${P[-1]} 2>/dev/null || break; sleep 5; done
done
for p in "\${P[@]}"; do wait \$p; echo "pid \$p exit \$?"; done
grep -l "ERROR" $B2C/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_* || true
EOF
  if [ $DRY = 1 ]; then echo "--dry-run：池 $pool $NT 个作业，作业脚本 $JOB"; return 0; fi
  local JID; JID=$(sbatch --parsable "$JOB")
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$JID" "$NT" "$MAN" "$pool" >> "$ST/submissions.tsv"
  echo "池 $pool：已提交数组作业 $JID（$NT 个任务，同时至多 $conc 个）"
}
submit_pool a100-40 1 "$A100_CONC" "$BASE_EXCLUDE_A100"
submit_pool h100-47 1 "$H47_CONC" "$BASE_EXCLUDE_H100" "$H100_NODES"
echo "跑完后重跑本脚本会只补缺的键；python formal_status.py 看进度（盲）。"
