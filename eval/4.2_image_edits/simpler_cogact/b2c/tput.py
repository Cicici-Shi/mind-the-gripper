"""吞吐汇总（只读 sec、gpu_name、host、block、cell，不读成败）：每个（目录、型号、块、格）的每回合秒数。
用法：python tput.py <目录:标签> ...   例：python tput.py out_smoke:A100x1 out_smoke_h100:H100x3
"""
import collections
import glob
import os
import sys

import numpy as np

rows = collections.defaultdict(list)
for arg in sys.argv[1:]:
    d, tag = arg.split(":")
    for p in glob.glob(os.path.join(d, "*", "*", "*.npz")):
        with np.load(p, allow_pickle=False) as z:
            rows[(tag, str(z["gpu_name"]), str(z["block"]), str(z["cell"]))].append(float(z["sec"]))
for k in sorted(rows):
    v = rows[k]
    print(f"{k[0]:12s} {k[1]:24s} {k[2]} {k[3]}  n={len(v):3d}  mean={np.mean(v):6.1f}  median={np.median(v):6.1f}  max={np.max(v):6.1f}")
