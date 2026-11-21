#!/bin/bash
# bench2 补充协议（填充颜色与更长滞后，protocol_bench2_ext_fill_lag.md）的正式采集驱动，b2x 版。
# 与 b2/submit_formal.sh 的差别：目录 b2x、清单 FROZEN_bench2_ext、输出 out_formal_ext、作业名 b2x_；
# 块与格从清单取（B2 四格、F1 八格）；每格的额外参数（填充颜色、C5L 的滞后、F1 的几何量滞后）也从清单取。
# 补充协议 v1 §1 的两步：
#   bash submit_formal.sh --stage1   第一步：每个旧格（C0、C1、C3、C4、C5）只跑前 10 个键；跑完用 cmp_phase1.py --stage1 比对，
#                                    与第一阶段记录全部逐位相同才写 formal_state/STAGE1_OK
#   bash submit_formal.sh            全量：要求 STAGE1_OK 存在；只为缺记录的（块，格，键）出作业（10 键已有的跳过）
#   加 --dry-run 只打印计划，不提交，不要求冻结标记。
# 可断点续跑：每次运行只为缺记录的（块，格，键）出作业。
#
#   bash submit_formal.sh             真提交：要求 FROZEN_bench2_ext 存在，且当前代码与键清单的哈希与它一致
#   bash submit_formal.sh --dry-run   空跑：不要求冻结标记，只打印计划，把切片和作业脚本生成到一个临时目录，不提交
#
# 冻结清单（json）：真提交用 FROZEN_bench2（协调方由 FROZEN_bench2.candidate 核对后改名，本脚本不创建它）；
#   空跑在 FROZEN_bench2 不存在时用 FROZEN_bench2.candidate。各块的 L、线宽都从清单取。
# 键清单：keysets/formal_<块>.json（哈希记在清单里）；真提交首次运行时复制成 formal_state/keys_<块>.json，之后不再改。
# 做法：缺的键按（块，格）切成至多 n_max 个、尽量等大的份，每 PAR 份装进一个 GPU 作业（同卡 PAR 个进程并行）；
#   作业以 slurm 数组提交，%2 限制同时最多 2 个 GPU 作业；每个作业 --time 不超过 60 分钟。
#   eval_views.py 带 --skip-done，记录先写临时文件再改名，所以重复提交或中途被杀都不会出半截记录。
# 状态落盘：formal_state/（键清单、分片、submissions.tsv 提交记录）。日志里不打印成败。
set -euo pipefail
B2=$PROJ_ROOT/bench2_simpler/b2x
FROZEN=$B2/FROZEN_bench2_ext
CAND=$B2/FROZEN_bench2_ext.candidate
OUT=$B2/out_formal_ext
BLOCKS=(B2 F1)
PAR=${PAR:-4}            # 每张卡并行进程数（吞吐作业 <job> 实测：4 进程时吞吐约为单进程的 3.7 倍，显存 32 GB 封顶）
CHUNK=${CHUNK:-}         # 每个进程的键数上限；不设则按块的每回合耗时估
DRY=0; STAGE1=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --stage1) STAGE1=1 ;;
    *) echo "未知参数 $arg" >&2; exit 1 ;;
  esac
done

cd "$B2"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
export PYTHONDONTWRITEBYTECODE=1

if [ $DRY = 0 ]; then
  if [ ! -f "$FROZEN" ]; then
    echo "拒绝提交：$FROZEN 不存在（协议未冻结）。" >&2; exit 1
  fi
  MANIFEST=$FROZEN
  ST=$B2/formal_state
  if [ $STAGE1 = 0 ] && [ ! -f "$ST/STAGE1_OK" ]; then
    echo "拒绝提交全量：$ST/STAGE1_OK 不存在（先 --stage1，再 python cmp_phase1.py --stage1 确认旧格逐位相同）。" >&2; exit 1
  fi
  # 防重复提交：先把 squeue 的输出完整取到变量里再查（管道 + grep -q 在 pipefail 下会因 SIGPIPE 误判为「没有」）
  if ! QJOBS=$(squeue -u "$USER" -h -o "%j"); then
    echo "拒绝提交：squeue 失败，无法确认没有在跑的 b2x_ 作业。" >&2; exit 1
  fi
  if [ -n "$(printf '%s\n' "$QJOBS" | awk '/^b2x_f/ {print; exit}')" ]; then
    echo "已有 b2x_f 作业在队列里，先等它们结束（避免同一键被两个作业同时跑）。" >&2; exit 1
  fi
else
  if [ -f "$FROZEN" ]; then MANIFEST=$FROZEN; else MANIFEST=$CAND; fi
  [ -f "$MANIFEST" ] || { echo "空跑也需要冻结清单或候选：$FROZEN / $CAND 都不存在（先跑 freeze_manifest.py）。" >&2; exit 1; }
  ST=$(mktemp -d "$B2/_dryrun_XXXXXX")
  echo "--dry-run：清单 $MANIFEST，临时目录 $ST（不提交）"
fi

# 核对当前代码与键清单的哈希和冻结清单一致（空跑也核，不一致时真提交拒绝、空跑报出）
python - "$MANIFEST" "$B2" "$DRY" <<'PY'
import hashlib, json, os, sys
man, b2, dry = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3] == "1"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f"代码 {f}" for f, h in man["code_sha256"].items() if sha(os.path.join(b2, f)) != h]
bad += [f"键清单 {b}" for b, h in man["keylist_sha256"].items() if sha(os.path.join(b2, "keysets", f"formal_{b}.json")) != h]
if bad:
    print(("空跑提示" if dry else "拒绝提交") + "：与冻结清单不一致：" + "，".join(bad), file=sys.stderr)
    sys.exit(0 if dry else 1)
print("代码与键清单的哈希与冻结清单一致", file=sys.stderr)
PY

mkdir -p "$ST/chunks"
[ $DRY = 1 ] || mkdir -p "$B2/logs/formal"
for b in "${BLOCKS[@]}"; do
  if [ ! -f "$ST/keys_$b.json" ]; then
    [ -f "$B2/keysets/formal_$b.json" ] || { echo "拒绝提交：缺 $B2/keysets/formal_$b.json（冻结的键清单）" >&2; exit 1; }
    cp "$B2/keysets/formal_$b.json" "$ST/keys_$b.json"
  fi
  cmp -s "$ST/keys_$b.json" "$B2/keysets/formal_$b.json" || { echo "拒绝提交：$ST/keys_$b.json 与冻结的键清单不一致" >&2; exit 1; }
done

TAG=$(date +%Y%m%d_%H%M%S)
MAN="$ST/manifest_$TAG.txt"   # 每行一个数组任务：空格分隔的「块:格:L:线宽:填充色:长滞后:几何滞后:平涂统计:分片文件」，至多 PAR 个；没有的参数写 -
python - "$ST" "$OUT" "$TAG" "$PAR" "${CHUNK:-0}" "$MAN" "$MANIFEST" "$STAGE1" <<'PY'
import json, os, sys
import numpy as np
st, out, tag, par, chunk, man, frozen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
stage1 = sys.argv[8] == "1"
sys.path.insert(0, os.getcwd())  # 脚本已 cd 到 b2x
import keys as K
blocks = json.load(open(frozen))["blocks"]
FILL_KEY = {"C1w": "w", "C1L": "L"}
FILLED = ("C1", "C1w", "C1L")
# 每回合秒数：同卡 4 进程时的实测值乘 1.2 到 1.3 的余量（协议 v1.1 冻结前，接手方按实测改）。
#   B1 22.3 s（吞吐作业 <job>，C4，4 进程）-> 27；F1 27.1 s（同上）-> 33；
#   B2 15.4 s（校准作业 <job>，C0，4 块并行）-> 20（C4、C5 每步多一次渲染，余量取大些）；F2 24.9 s（同上）-> 33（按 F1 的 C4 取）。
# 每个进程至多留 47 分钟跑回合（作业 58 分钟，扣载入约 40 秒与余量）；键数按「至多 n_max、各份尽量一样多」切。
SEC = {"B1": 27, "B2": 20, "F1": 33, "F2": 33}
tasks = []
for b in ("B2", "F1"):
    keys = json.load(open(os.path.join(st, f"keys_{b}.json")))
    n_max = chunk or max(1, int((47 * 60) // SEC[b]))
    mb = blocks[b]
    L, lw = int(mb["L"]), int(mb["line_width"])
    # F1 的每格都记 4 步与更长滞后两种几何量（§2：正式回合里再照报一遍）；B2 不涉及滞后
    geom = f"{int(mb['cell_lag']['C5'])},{int(mb['cell_lag']['C5L'])}" if "C5L" in mb["cell_lag"] else "-"
    for c in mb["cells"]:
        if stage1 and c not in K.EXT_OLD_CELLS:
            continue
        fill = ",".join(str(int(x)) for x in mb["fill_rgb"][FILL_KEY[c]]) if c in FILL_KEY else "-"
        lagl = str(int(mb["cell_lag"]["C5L"])) if c == "C5L" else "-"
        fstat = "1" if c in FILLED else "-"
        d = os.path.join(out, b, c)
        miss = []
        for k in (keys[:K.EXT_STAGE1_N] if stage1 else keys):
            p = os.path.join(d, k["key"] + ".npz")
            ok = False
            if os.path.exists(p):
                try:
                    with np.load(p) as z:
                        ok = bool(z["complete"])
                except Exception:
                    ok = False
            if not ok:
                miss.append(k)
        n_chunk = -(-len(miss) // n_max) if miss else 0
        n = -(-len(miss) // n_chunk) if n_chunk else 1
        for i in range(0, len(miss), n):
            f = os.path.join(st, "chunks", f"{tag}_{b}_{c}_{i // n:03d}.json")
            with open(f, "w") as fh:
                json.dump(miss[i:i + n], fh)
            tasks.append(f"{b}:{c}:{L}:{lw}:{fill}:{lagl}:{geom}:{fstat}:{f}")
        print(f"{b} {c}: 缺 {len(miss)} 键，{n_chunk} 份", file=sys.stderr)
with open(man, "w") as fh:
    for i in range(0, len(tasks), par):
        fh.write(" ".join(tasks[i:i + par]) + "\n")
print(f"共 {len(tasks)} 份", file=sys.stderr)
PY
NT=$(wc -l < "$MAN" | tr -d ' ')
if [ "$NT" = "0" ]; then echo "没有缺的键，全部完成。"; exit 0; fi
echo "本次需要 $NT 个 GPU 作业（每个至多 $PAR 个进程），清单 $MAN"
JOB=$ST/job_$TAG.sbatch
# 内存：吞吐作业 <job> 同卡 4 进程（权重 mmap）实测 MaxRSS 75 GB；64G 时 5 进程的冒烟 <job> 被 OOM 杀过，所以给 128G
cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH -J b2x_f_$TAG
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:58:00
#SBATCH --array=0-$((NT - 1))%2
#SBATCH -o $B2/logs/formal/%x_%A_%a.log
source $B2/jobs/common.sh
# 只给 Vulkan 加载器 nvidia 的 ICD（common.sh 里也设了；这里再显式设一次，值写进每条记录，门槛 4 核对）
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" $MAN)
for t in \$LINE; do
  IFS=: read b c L lw fill lagl geom fstat f <<< "\$t"
  extra=""
  [ "\$fill" != "-" ] && extra="\$extra --fill-rgb \$fill"
  [ "\$lagl" != "-" ] && extra="\$extra --lag-long \$lagl"
  [ "\$geom" != "-" ] && extra="\$extra --geom-lags \$geom"
  [ "\$fstat" != "-" ] && extra="\$extra --fill-stats"
  python eval_views.py --block \$b --cell \$c --keys \$f --out $OUT --lag \$L --line-width \$lw --skip-done \$extra \
    > $B2/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${b}_\${c}_\$(basename \$f .json).log 2>&1 &
done
wait
grep -l "ERROR" $B2/logs/formal/run_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_* || true
EOF
if [ $DRY = 1 ]; then echo "--dry-run：不提交。作业脚本 $JOB"; exit 0; fi
JID=$(sbatch --parsable "$JOB")
printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$JID" "$NT" "$MAN" >> "$ST/submissions.tsv"
echo "已提交数组作业 $JID（$NT 个任务，同时至多 2 个）。跑完后重跑本脚本会只补缺的键。"
