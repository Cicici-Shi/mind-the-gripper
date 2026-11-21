"""第二阶段环境不变检查（由第一阶段 b2/cmp_smoke.py 改写，多比一项 pred_chunks，即每次查询的扩散输出）：
两次冒烟（同一组校准键、五格）逐条比 raw_actions、pred_chunks、env_actions、frame0_sha256 是否逐位相同。
只打印相同条数与不同的（块，格，键），绝不读、不打印成败字段。
用法：python cmp_smoke.py --new out_smoke_post --ref out_smoke
"""
import argparse
import glob
import os
import sys

import numpy as np

FIELDS = ("raw_actions", "pred_chunks", "env_actions", "frame0_sha256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    ref = sorted(os.path.relpath(p, a.ref) for p in glob.glob(os.path.join(a.ref, "*", "*", "*.npz")))
    same = {f: 0 for f in FIELDS}
    all_same, missing, diff = 0, [], []
    for rel in ref:
        pn = os.path.join(a.new, rel)
        if not os.path.exists(pn):
            missing.append(rel)
            continue
        with np.load(os.path.join(a.ref, rel), allow_pickle=False) as zr, np.load(pn, allow_pickle=False) as zn:
            ok = {}
            for f in FIELDS:
                x, y = zr[f], zn[f]
                ok[f] = x.shape == y.shape and x.dtype == y.dtype and bool(np.array_equal(x, y))
                same[f] += ok[f]
        if all(ok.values()):
            all_same += 1
        else:
            diff.append((rel[:-4], [f for f in FIELDS if not ok[f]]))
    print(f"参照记录 {len(ref)} 条；新目录缺 {len(missing)} 条")
    print("逐字段相同条数：" + "，".join(f"{f} {same[f]}/{len(ref)}" for f in FIELDS))
    print(f"三项全部逐位相同：{all_same}/{len(ref)}")
    for rel, fs in diff:
        print(f"  不同：{rel} {fs}")
    for rel in missing:
        print(f"  缺：{rel}")
    print("ENV_UNCHANGED" if all_same == len(ref) and ref else "ENV_CHANGED_OR_INCOMPLETE")
    return 0 if all_same == len(ref) and ref else 1


if __name__ == "__main__":
    sys.exit(main())
