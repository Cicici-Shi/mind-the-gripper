import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pg_smoke as S
runs = {w: S.rollout("erase", w) for w in ("none", "all", "grasp", "transport")}
for w in ("grasp", "transport"):
    r = runs[w]
    print("RUN", w, "gates", "".join("1" if g else "0" for g in r["gates"]))
    for k in range(len(r["gates"])):
        dn = S.ndiff(r["frames"][k], runs["none"]["frames"][k]); da = S.ndiff(r["frames"][k], runs["all"]["frames"][k])
        print(f"DBG {w} k={k} gate={int(r['gates'][k])} vs_none={dn} vs_all={da} none_vs_all={S.ndiff(runs['none']['frames'][k], runs['all']['frames'][k])}")
