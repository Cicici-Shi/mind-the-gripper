"""Compare per-episode fixture positions logged by pi0.5 v2 runs with the OFT SRC dumps."""
import glob
import json
import sys

import numpy as np

R = "$PROJ_ROOT/runs/src"
worst, n = 0.0, 0
for f in sorted(glob.glob("$PROJ_ROOT/pi05/v2/*_e*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        g = glob.glob("%s/t%d_e%d_i*.npz" % (R, r["task"], r["episode"]))
        if not g:
            print("no OFT dump", f, r["task"], r["episode"])
            continue
        z = np.load(g[0], allow_pickle=True)
        names = [str(x) for x in z["ref_body_names"]]
        d = [np.abs(np.array(v) - z["ref_body_pos"][names.index(k)]).max() * 1000 for k, v in r["fix"].items() if k in names]
        oft_init = int(g[0].split("_i")[-1][:-4])
        n += 1
        worst = max(worst, max(d))
        if max(d) > 0.01 or oft_init != r["init"] or "-v" in sys.argv:
            print(f.split("/")[-1], "task", r["task"], "ep", r["episode"], "init", r["init"], "oft_init", oft_init, "max diff mm %.3f" % max(d), "bodies", len(d))
print("episodes checked", n, "worst fixture diff mm %.4f" % worst)
