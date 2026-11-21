"""一次性脚本：修订 R1（2026-11-14）裁定五第 4 步。按原清单（FROZEN_bench2_cogact.orig_5ef81795）的分配，
把分给 H100 NVL 的键的 out_formal/<块>/<格>/<键>.npz（完整与否都算）与残留临时文件 .<键>.tmp.npz
原样移到 out_formal_void_h100/ 下同样的子目录，写作废清单 formal_state/void_h100_r1.tsv，作废目录设为只读。
全程盲：每个记录只读 complete 与 gpu_name 两个字段（np.load 惰性读取，只解压这两个成员）。
用法：python void_h100_r1.py [--do]   不带 --do 只盘点不移动。本文件不在清单哈希里。
"""
import collections
import hashlib
import json
import os
import shutil
import stat
import sys

import numpy as np

B2C = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(B2C, "FROZEN_bench2_cogact.orig_5ef81795")
SRC, DST = os.path.join(B2C, "out_formal"), os.path.join(B2C, "out_formal_void_h100")
TSV = os.path.join(B2C, "formal_state", "void_h100_r1.tsv")
CELLS = ("C0", "C1", "C3", "C4", "C5")
H100, A100 = "NVIDIA H100 NVL", "NVIDIA A100-PCIE-40GB"


def peek(p):
    try:
        with np.load(p, allow_pickle=False) as z:
            return (str(z["gpu_name"]) if "gpu_name" in z.files else "?",
                    bool(z["complete"]) if "complete" in z.files else False)
    except Exception as e:  # noqa: BLE001
        return f"读不了：{type(e).__name__}", False


def main():
    do = "--do" in sys.argv
    man = json.load(open(ORIG))
    rows, keys_hit = [], set()
    for b, amap in man["gpu_assignment"].items():
        for c in CELLS:
            d = os.path.join(SRC, b, c)
            for k, m in amap.items():
                if m != H100:
                    continue
                for name in (k + ".npz", f".{k}.tmp.npz"):
                    p = os.path.join(d, name)
                    if os.path.exists(p):
                        g, comp = peek(p)
                        rows.append((os.path.relpath(p, B2C), os.path.getsize(p),
                                     hashlib.sha256(open(p, "rb").read()).hexdigest(), g, comp, b, c, k))
                        keys_hit.add((b, k))
    n_npz = sum(1 for r in rows if not os.path.basename(r[0]).startswith("."))
    print(f"待作废：文件 {len(rows)}（记录 {n_npz}，临时 {len(rows) - n_npz}），键 {len(keys_hit)}，"
          f"按块 {dict(collections.Counter(r[5] for r in rows))}，型号 {dict(collections.Counter(r[3] for r in rows))}，"
          f"完整 {sum(r[4] for r in rows)}")
    per_key = collections.Counter((r[5], r[7]) for r in rows)
    print("每键文件数分布：", dict(collections.Counter(per_key.values())))
    if not do:
        return
    if os.path.exists(TSV):
        sys.exit(f"拒绝：{TSV} 已存在")
    with open(TSV, "w") as f:
        f.write("path\tbytes\tsha256\tgpu_name\tcomplete\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{int(r[4])}\n")
    for r in rows:
        src = os.path.join(B2C, r[0])
        dst = os.path.join(DST, os.path.relpath(src, SRC))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        assert not os.path.exists(dst), dst
        shutil.move(src, dst)
        assert hashlib.sha256(open(dst, "rb").read()).hexdigest() == r[2], dst
    for root, dirs, files in os.walk(DST):
        for x in files + dirs:
            p = os.path.join(root, x)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(DST, os.stat(DST).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(TSV, 0o444)
    # 验收：out_formal 剩下的
    left = collections.Counter()
    for b in man["gpu_assignment"]:
        for c in CELLS:
            d = os.path.join(SRC, b, c)
            for name in os.listdir(d):
                if name.endswith(".npz"):
                    g, comp = peek(os.path.join(d, name))
                    left[(g, comp)] += 1
    print("移走后 out_formal 剩下（型号, 完整）：", dict(left))


if __name__ == "__main__":
    main()
