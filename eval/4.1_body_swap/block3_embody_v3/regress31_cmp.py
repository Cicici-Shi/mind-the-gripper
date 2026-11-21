"""Compare old-code and new-code replays from tools/regress.py (2026-10-31): per pair, every episode's replayed qpos
trajectory and twin pad-residual series must be bit-identical (max |diff| == 0). Success flags are NOT printed.
Usage: python tools/regress31_cmp.py DIR pair_prefix [pair_prefix ...]   (expects DIR/<p>_old and DIR/<p>_new)"""
import glob
import json
import os
import sys

import numpy as np

d = sys.argv[1]
for p in sys.argv[2:]:
    a, b = os.path.join(d, p + "_old"), os.path.join(d, p + "_new")
    fa = sorted(glob.glob(os.path.join(a, "t*_e*.npz")))
    n = worst = worst_tw = 0
    lens_ok, missing, extra_keys = True, [], set()
    for f in fa:
        g = os.path.join(b, os.path.basename(f))
        if not os.path.exists(g):
            missing.append(os.path.basename(f))
            continue
        za, zb = np.load(f), np.load(g)
        if za["qpos"].shape != zb["qpos"].shape:
            lens_ok = False
            continue
        worst = max(worst, float(np.abs(za["qpos"] - zb["qpos"]).max()) if za["qpos"].size else 0.0)
        if "twin_pos" in za.files:
            worst_tw = max(worst_tw, float(np.abs(za["twin_pos"] - zb["twin_pos"]).max()))
        extra_keys |= set(zb.files) - set(za.files)
        n += 1
    ra = {(r["task"], r["ep"]): r for r in map(json.loads, open(os.path.join(a, "result.jsonl")))} if os.path.exists(os.path.join(a, "result.jsonl")) else {}
    rb = {(r["task"], r["ep"]): r for r in map(json.loads, open(os.path.join(b, "result.jsonl")))} if os.path.exists(os.path.join(b, "result.jsonl")) else {}
    same_flag = sum(ra[k]["replay_success"] == rb[k]["replay_success"] for k in ra if k in rb)
    same_steps = sum(ra[k]["steps"] == rb[k]["steps"] for k in ra if k in rb)
    cont = [rb[k].get("contacts") for k in sorted(rb)]
    fg = [c.get("first_grip_obj") for c in cont if c]
    print(f"{p:8s} episodes={n} qpos_max_diff={worst:.3g} twin_pos_max_diff={worst_tw:.3g} shapes_equal={lens_ok} "
          f"missing={missing} replay_flag_same={same_flag}/{len(ra)} steps_same={same_steps}/{len(ra)} "
          f"new_fields={sorted(extra_keys)} contacts_logged={sum(c is not None for c in cont)} first_grip_obj={fg[:10]}")
