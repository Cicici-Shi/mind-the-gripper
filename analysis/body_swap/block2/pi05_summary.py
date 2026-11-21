"""pi0.5 v2 summary: success per condition and paired difference vs SRC (clean records only)."""
import glob
import json
import sys

import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "$PROJ_ROOT/pi05/v2"


def load(c):
    d = {}
    for f in glob.glob(f"{P}/{c}_e*.jsonl"):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r["error"] is None:
                    d[(r["task"], r["episode"])] = float(r["success"])
    return d


src = load("src")
rng = np.random.default_rng(0)
print("cond             n    SR%   vs SRC pp [95% CI]")
for c in ["src", "sham", "rec", "appshid", "iiwa", "iiwahid", "ur5e", "ur5ehid", "grp", "grpkp100",
          "sham_twin", "iiwa_twin", "ur5e_twin", "grpkp100_twin", "ur5e_rev"]:
    d = load(c)
    ks = sorted(set(d) & set(src))
    if not ks:
        print(c, "none")
        continue
    a = np.array([d[k] for k in ks])
    diff = a - np.array([src[k] for k in ks])
    bs = [diff[rng.integers(0, len(ks), len(ks))].mean() * 100 for _ in range(5000)]
    print(f"{c:14s} {len(ks):4d} {100 * a.mean():6.1f}  {100 * diff.mean():+6.1f} [{np.percentile(bs, 2.5):+6.1f}, {np.percentile(bs, 97.5):+6.1f}]")
