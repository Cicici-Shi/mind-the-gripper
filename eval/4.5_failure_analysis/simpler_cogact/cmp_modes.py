"""比较 rgbd 与 state_dict 两种模式的回放输出是否逐位相同（tcp、actors、links、qpos、success_steps）。"""
import os, sys
import numpy as np
A, B = sys.argv[1], sys.argv[2]
n = same = 0
for root, _, files in os.walk(A):
    for f in files:
        if not f.endswith(".npz") or f.startswith("."):
            continue
        pa = os.path.join(root, f); pb = pa.replace(A, B, 1)
        if not os.path.exists(pb):
            print("missing", pb); continue
        za, zb = np.load(pa), np.load(pb)
        ok = all(np.array_equal(za[k], zb[k]) for k in ("tcp", "actors", "links", "qpos", "success_steps", "grasp_src"))
        n += 1; same += int(ok)
        if not ok:
            print("DIFF", pa)
print(f"[cmp] pairs {n} identical {same}")
