"""embody_next: summarise tools/nx_check.py outputs (no success field exists in them).
Usage: python tools/nx_summary.py DIR [DIR ...]   (each DIR = one body / cell, holding <suite>_<body>_e<off>.jsonl)
Per suite x scene type x mode: keys, init ok, natural / fixture / object diffs, base offset, init penetration,
breakdown rate (detector D + QACC + time resets), arm-scene contact pairs, twin IK residual distribution."""
import collections
import glob
import json
import os
import sys

import numpy as np

SCENE = {"libero_tabletop_manipulation": "Tabletop", "libero_kitchen_tabletop_manipulation": "Kitchen",
         "libero_living_room_tabletop_manipulation": "LivingRoom", "libero_study_tabletop_manipulation": "Study"}


def scene_of(r):
    if r.get("scene"):
        return SCENE.get(r["scene"][2], r["scene"][2])
    if r["suite"] == "libero_goal":
        return "Tabletop"
    return "Kitchen" if r["task"] <= 3 else ("LivingRoom" if r["task"] <= 8 else "Study")


for d in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))) for l in open(f)]
    if not rows:
        print(d, "no rows")
        continue
    print(f"##### {d}  rows {len(rows)}  body {rows[0]['body']}  env {rows[0].get('cell_env')}")
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["suite"], scene_of(r), r["mode"])].append(r)
    for k in sorted(grp):
        R = grp[k]
        first = [r for r in R if "natural_mm" in r]
        def mx(key, rs=R, sub=None):
            v = [(r[key] if sub is None else (r[key] or {}).get(sub)) for r in rs if r.get(key) is not None]
            v = [x for x in v if isinstance(x, (int, float))]
            return round(max(v), 4) if v else None
        bd = [r for r in R if r["breakdown"]]
        line = (f"{k[0]:14s} {k[1]:10s} {k[2]:6s} keys {len(R):3d} init_ok {sum(r['init_ok'] for r in R)}/{len(R)}"
                f" natural_mm {mx('natural_mm', first)} natural_si_mm {mx('natural_si_mm', first)} fixture_mm {mx('fixture_mm')}"
                f" obj_free_mm {max(r['objects']['free_mm'] for r in R):.4f} obj_art {max(r['objects']['articulated_abs'] for r in R):.2e}"
                f" base_mm {max((max(abs(x) for x in r['base_mm']) for r in R if r.get('base_mm')), default=None)}"
                f" align_pos_mm {max((r['align']['pos_mm'] for r in R if r.get('align')), default=None)}"
                f" align_rot_deg {max((r['align']['rot_deg'] for r in R if r.get('align')), default=None)}")
        print(line)
        pen = [r["init_pen_mm"] for r in R]
        print(f"      init penetration: >1mm {sum(p > 1 for p in pen)} max {max(pen):.2f} mm;"
              f" breakdown {len(bd)}/{len(R)} (D_arm {sum(bool(r['D_arm']) for r in R)}, D_finger {sum(bool(r['D_finger']) for r in R)},"
              f" D_slide {sum(bool(r.get('D_slide')) for r in R)}, qacc {sum(r['qacc'] > 0 for r in R)}, resets {sum(r['time_resets'] > 0 for r in R)},"
              f" loose {sum(r['D_loose'] for r in R)}) keys {[(r['task'], r['ep']) for r in bd][:12]}")
        if any("src_qpos_maxdiff" in r for r in R):
            print(f"      SRC replay vs dump qpos max {max(r['src_qpos_maxdiff'] for r in R if 'src_qpos_maxdiff' in r):.3e}")
        ip = collections.Counter()
        for r in R:
            for p in r["init_pairs"]:
                ip[p] += 1
        if ip:
            print(f"      init contact pairs (keys): {dict(ip.most_common(6))}")
        cp = collections.Counter()
        cs = [r["contacts"]["arm_env_contact_steps"] for r in R if r.get("contacts")]
        for r in R:
            for p in (r.get("contacts") or {}).get("pairs", {}):
                cp[p.split("|")[1]] += 1
        if cs:
            print(f"      arm-scene contact: keys with any {sum(c > 0 for c in cs)}/{len(cs)}, steps median {np.median(cs):.0f} max {max(cs)};"
                  f" objects touched by arm (keys) {dict(cp.most_common(6))}")
        tw = [r["twin"] for r in R if r.get("twin")]
        if tw:
            fr = sum(t["n"] for t in tw)
            print(f"      twin: frames {fr}, >1mm {sum(t['gt1mm'] for t in tw)} ({100 * sum(t['gt1mm'] for t in tw) / fr:.2f}%),"
                  f" >2mm {sum(t['gt2mm'] for t in tw)}, query frames >1mm {sum(t['q_gt1mm'] for t in tw)}/{sum(t['nq'] for t in tw)},"
                  f" per-key p50 median {np.median([t['pos_p50'] for t in tw]):.4f} p99 max {max(t['pos_p99'] for t in tw):.3f}"
                  f" max {max(t['pos_max'] for t in tw):.3f} mm, rot max {max(t['rot_max'] for t in tw):.3f} deg,"
                  f" keys with >1mm {sum(t['gt1mm'] > 0 for t in tw)}, retries {sum(t['retries'] for t in tw)}, stale {sum(t['stale'] for t in tw)},"
                  f" cam_err max {max(t['cam_err_mm_max'] for t in tw):.4f} mm / {max(t['cam_err_deg_max'] for t in tw):.4f} deg,"
                  f" t0 agent_mean max {max(t['t0']['agentview_image'][1] for t in tw):.3f} wrist_mean max {max(t['t0']['robot0_eye_in_hand_image'][1] for t in tw):.3f}")
            bad = [(r["task"], r["ep"], r["twin"]["gt1mm"], round(r["twin"]["pos_max"], 1), bool(r["breakdown"])) for r in R if r.get("twin") and r["twin"]["gt1mm"]]
            if bad:
                print(f"      twin keys with >1mm (task, ep, frames, max mm, breakdown): {bad[:15]}")
