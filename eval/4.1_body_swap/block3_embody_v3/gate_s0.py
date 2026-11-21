"""Shard gate check for block-3 cells (protocol_block3 4.1 / 4.2, amendment 1 rulings 1-3; written 2026-10-31).

Prints, per cell and model, only gate quantities -- never a success rate:
  n episodes (keys present / expected), pi0.5 error records, copy_grip_cmd flags (all must be 1),
  first-frame alignment max pos_mm / rot_deg (gate 0.01 mm / 0.05 deg) and fixture_mm, scene identity against the
  OFT src dump of the same key (gate 0.0000 mm; pi0.5 'fix' is rounded to 1e-5 m, so its tolerance is 0.01 mm),
  online contact log present, twin query-frame IK misses (pos > 2 mm or rot > 1 deg; G_R: pad residual > 1 mm),
  stale frames, same-key agreement with the paired cell (twin cells only), UR5e_nc forearm penetration count on
  task 3, G_R root-offset summary (lateral = hypot(dx, dy), before / after the first gripper-object contact).

Usage: python tools/gate_s0.py --model oft|pi05 --cell NAME [--pair NAME] [--offsets 0] [--gr]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=("oft", "pi05"), required=True)
ap.add_argument("--cell", required=True)
ap.add_argument("--pair", default="")
ap.add_argument("--offsets", default="0")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--gr", action="store_true", help="G_R: pad-residual gate and root offsets")
ap.add_argument("--gr-t0", default="", help="dir of gr_t0check jsonl: amendment 2 ruling 2 (D2) hidden-gripper cell means")
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--pi", default=os.path.expanduser("$PROJ_ROOT/pi05/v2"))
args = ap.parse_args()
OFFS = [int(x) for x in args.offsets.split(",")]
KEYS = [(t, e) for o in OFFS for t in range(10) for e in range(o, o + args.trials)]
ALIGN_RE = re.compile(r"BODY_ALIGN: body=(\S+) pos_mm=([\d.]+) rot_deg=([\d.]+) quat_sign=\S+ fixture_mm=([\d.]+) "
                      r"missing=(\[.*?\])(?:.*?copy_grip_cmd=(\d))?")


def oft_eps(cell):
    out = {}
    for f in glob.glob(os.path.join(args.runs, cell, "t*_e*_i*.npz")):
        m = re.search(r"t(\d+)_e(\d+)_i", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        if k in KEYS:
            out[k] = f
    return out


def oft_align_from_txt(note):
    """BODY_ALIGN lines of the EVAL txt logs of this NOTE, keyed by the preceding INIT_ID line."""
    res = {}
    for f in sorted(glob.glob(os.path.expanduser(f"$PROJ_ROOT/openvla-oft/experiments/logs/EVAL-libero_spatial-*--{note}.txt"))):
        key = None
        for line in open(f, errors="replace"):
            m = re.search(r"INIT_ID: task=(\d+) episode=(\d+)", line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
            m = ALIGN_RE.search(line)
            if m and key is not None:
                res[key] = dict(pos_mm=float(m.group(2)), rot_deg=float(m.group(3)), fixture_mm=float(m.group(4)),
                                missing=m.group(5), copy_grip_cmd=int(m.group(6)) if m.group(6) else -1, src=os.path.basename(f))
    return res


def pi_recs(cell):
    recs, errs = {}, 0
    for f in glob.glob(os.path.join(args.pi, f"{cell}_e*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["task"], r["episode"])
            if k not in KEYS:
                continue
            if r.get("error") is not None:
                errs += 1
                continue
            recs.setdefault(k, r)
    return recs, errs


def pi_align_from_logs(cell):
    """Client logs: BODY_ALIGN line printed before the episode's EP line."""
    res = {}
    for f in glob.glob(os.path.expanduser(f"$PROJ_ROOT/pi05/logs/v2_{cell}_*_e*.log")):
        pend = None
        for line in open(f, errors="replace"):
            m = ALIGN_RE.search(line)
            if m:
                pend = dict(pos_mm=float(m.group(2)), rot_deg=float(m.group(3)), fixture_mm=float(m.group(4)),
                            missing=m.group(5), copy_grip_cmd=int(m.group(6)) if m.group(6) else -1)
            if line.startswith("EP {") and pend is not None:
                r = json.loads(line[3:line.rindex("}") + 1])
                res[(r["task"], r["episode"])] = pend
                pend = None
    return res


def src_fix(k):
    fs = glob.glob(os.path.join(args.runs, "src", f"t{k[0]}_e{k[1]}_i*.npz"))
    if not fs:
        return None
    z = np.load(fs[0], allow_pickle=True)
    return {str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])}


def fixdiff(names_pos, ref):
    worst = 0.0
    for n, p in names_pos.items():
        if not n or n == "world" or n.startswith(("robot0", "gripper0", "mount0")) or n not in ref:
            continue
        worst = max(worst, float(np.abs(np.asarray(p, float) - ref[n]).max()) * 1000)
    return worst


def summ(x):
    x = np.asarray(x, float)
    if not x.size:
        return "n=0"
    return f"n={x.size} median={np.median(x):.3f} p95={np.percentile(x, 95):.3f} max={x.max():.3f}"


print(f"== {args.model} {args.cell} offsets={OFFS} expected={len(KEYS)}")
succ, align, fixmax, contacts_ok, contacts_err = {}, {}, 0.0, 0, 0
twin_q = twin_miss = twin_stale = 0
pen3 = []
root_pre, root_post, root_pre_flag = [], [], 0
if args.model == "oft":
    eps = oft_eps(args.cell)
    txt_align = oft_align_from_txt(args.cell)
    hosts = set()
    for k, f in sorted(eps.items()):
        z = np.load(f, allow_pickle=True)
        succ[k] = bool(z["success"])
        a = json.loads(str(z["align_json"])) if "align_json" in z.files else txt_align.get(k)
        align[k] = a
        if "gpu" in z.files:
            hosts.add((str(z["host"]), str(z["gpu"])))
        ref = src_fix(k)
        if ref is not None:
            fixmax = max(fixmax, fixdiff({str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])}, ref))
        if "contact_arm_env_contact_steps" in z.files:
            contacts_ok += 1
        if "contact_error" in z.files:
            contacts_err += 1
        if "contact_forearm_pen_steps" in z.files and k[0] == 3:
            pen3.append(int(z["contact_forearm_pen_steps"]))
        if "twin_pos_err_mm" in z.files:
            pe, re_ = z["twin_pos_err_mm"][::8], z["twin_rot_err_deg"][::8]
            twin_q += len(pe)
            twin_miss += int(((pe > 1) if args.gr else ((pe > 2) | (re_ > 1))).sum())
            twin_stale += int(z["twin_stale"])
        if args.gr and "twin_root_off_mm" in z.files:
            ro = z["twin_root_off_mm"][::8]
            lat = np.hypot(ro[:, 0], ro[:, 1])
            fg = int(z["contact_first_grip_obj"]) if "contact_first_grip_obj" in z.files else -1
            steps = np.arange(len(z["twin_root_off_mm"]))[::8]
            pre = lat[(steps <= fg) if fg >= 0 else np.ones_like(lat, bool)]
            post = lat[steps > fg] if fg >= 0 else lat[:0]
            root_pre += list(pre)
            root_post += list(post)
            root_pre_flag += int((pre > 1).sum())
    print(f"hosts/gpus in dumps: {sorted(hosts) or 'not recorded (pre-2026-10-31 code): see slurm log first line'}")
else:
    recs, errs = pi_recs(args.cell)
    log_align = pi_align_from_logs(args.cell)
    print(f"pi0.5 error records dropped: {errs}")
    hosts = set()
    for k, r in sorted(recs.items()):
        succ[k] = bool(r["success"])
        align[k] = r.get("align") or log_align.get(k)
        if r.get("gpu"):
            hosts.add((r["host"], r["gpu"]))
        ref = src_fix(k)
        if ref is not None and r.get("fix"):
            fixmax = max(fixmax, fixdiff(r["fix"], ref))
        c = r.get("contacts")
        if c and "error" not in c:
            contacts_ok += 1
            if k[0] == 3 and "forearm_pen_steps" in c:
                pen3.append(int(c["forearm_pen_steps"]))
        elif c:
            contacts_err += 1
        tw = r.get("twin")
        if tw:
            q = tw.get("pos_err_mm_q") or []
            if args.gr:  # G_R: pad residual at query frames
                twin_q += len(q)
                twin_miss += int(sum(1 for a in q if a > 1))
            else:  # IK cells: pi0.5 records misses over ALL painted frames (no per-query rotation), as block 2
                twin_q += int(tw["n_paint"])
                twin_miss += int(tw["miss"])
            twin_stale += int(tw["stale"])
            if args.gr and tw.get("root_off_mm_q"):
                ro = np.asarray(tw["root_off_mm_q"], float)
                lat = np.hypot(ro[:, 0], ro[:, 1])
                fg = int((c or {}).get("first_grip_obj", -1))
                steps = np.arange(len(ro)) * 5
                pre = lat[(steps <= fg) if fg >= 0 else np.ones_like(lat, bool)]
                root_pre += list(pre)
                root_post += list(lat[steps > fg] if fg >= 0 else [])
                root_pre_flag += int((pre > 1).sum())
    print(f"hosts/gpus in records: {sorted(hosts) or 'not recorded: see job log first line'}")

missing = [k for k in KEYS if k not in succ]
al = [a for a in align.values() if a]
print(f"episodes present {len(succ)}/{len(KEYS)}; missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
if al:
    cg = [int(a.get("copy_grip_cmd", -1)) for a in al]
    pos = max(float(a["pos_mm"]) for a in al)
    rot = max(float(a["rot_deg"]) for a in al)
    fx = max(float(a["fixture_mm"]) for a in al)
    miss_obj = sum(1 for a in al if str(a.get("missing")) not in ("[]", "None"))
    newbody = args.cell in ("p6j3", "k3", "k3_twin", "rt2", "rt2_twin", "ur5e_nc", "obj_k3")  # amendment 2 ruling 1 (D1)
    nrot = sum(float(a["rot_deg"]) > 0.05 for a in al)
    print(f"align info for {len(al)} eps: copy_grip_cmd=1 in {cg.count(1)}/{len(cg)} (GATE all) | max pos_mm={pos:.4f} "
          f"max rot_deg={rot:.4f} (> 0.05 deg: {nrot} eps) {'(GATE 0.01 mm / 0.05 deg: new body)' if newbody else '(NA: body accepted in block 1/2, amendment 2 ruling 1; printed only)'} "
          f"| max fixture shift applied={fx:.3f} mm (not a gate) | eps with missing objects: {miss_obj}")
else:
    print("align info: NONE FOUND")
print(f"scene vs OFT src same key: max {fixmax:.4f} mm (GATE 0.0000; pi0.5 record precision 0.01 mm)")
print(f"online contact log present {contacts_ok}/{len(succ)}; contact_error {contacts_err}")
if twin_q:
    print(f"twin {'all painted' if (args.model == 'pi05' and not args.gr) else 'query'} frames {twin_q}, {'pad residual > 1 mm' if args.gr else 'IK misses'}: {twin_miss} "
          f"({100 * twin_miss / twin_q:.2f}%, GATE < 1%) | stale {twin_stale} (GATE 0)")
if pen3:
    print(f"UR5e_nc task 3: episodes with forearm penetration {sum(p > 0 for p in pen3)}/{len(pen3)} (full-cell check needs >= 15/20)")
if args.gr:
    print(f"G_R root lateral offset at query frames, before first gripper-object contact: {summ(root_pre)}; "
          f"frames > 1 mm before contact: {root_pre_flag} (expect 0; >0 = stop and check)")
    print(f"G_R root lateral offset after contact: {summ(root_post)}; all: {summ(root_pre + root_post)} (P9 descriptive if p95 > 6.4)")
if args.pair:
    if args.model == "oft":
        ps = {k: bool(np.load(f, allow_pickle=True)["success"]) for k, f in oft_eps(args.pair).items()}
    else:
        ps = {k: bool(r["success"]) for k, r in pi_recs(args.pair)[0].items()}
    common = [k for k in succ if k in ps]
    agree = sum(succ[k] == ps[k] for k in common)
    print(f"same-key agreement with {args.pair}: {agree}/{len(common)} = {100 * agree / max(len(common), 1):.1f}% (GATE >= 90%, else look at images)")

if args.gr_t0:  # amendment 2 ruling 2 (D2): hidden-gripper means per job and overall, gate < 0.11 each; others descriptive
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(args.gr_t0, "*.jsonl"))) for l in open(f) if l.strip()]
    by = {}
    for r in R:
        by.setdefault(f"{r.get('host')}/{r.get('gpu')}", []).append(r)
    for nm, rs in [("ALL", R)] + sorted(by.items()):
        h = [r.get("gr_vs_t0_hidden") or {} for r in rs]
        if any(x.get("wrist_mean") is None for x in h):
            print(f"gr_t0 {nm}: missing hidden fields -> UNK"); continue
        am, wm = np.mean([x["agent_mean"] for x in h]), np.mean([x["wrist_mean"] for x in h])
        sw = [r["gr_vs_t0"]["wrist_mean_outside_mask"] for r in rs]
        print(f"gr_t0 {nm} ({len(rs)} keys): hidden mean agent {am:.4f} wrist {wm:.4f} (GATE < 0.11) {'PASS' if max(am, wm) < 0.11 else 'FAIL'} | "
              f"hidden wrist per-key max {max(x['wrist_mean'] for x in h):.3f}; segmentation wrist mean {np.mean(sw):.3f} max {max(sw):.3f} (descriptive)")
