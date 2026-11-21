"""maskinfo-confirm gates (protocol_maskinfo_confirm.md v1 §1 / §4: the eight items of protocol_maskinfo.md §4 unchanged,
plus item 9 key-set check, item 10 label fields; item 11 is a WORKLOG entry) -> gate json read by
prereg_maskinfo_confirm.py. Copy of the frozen tools/gate_maskinfo.py (sha256 3ffe63bf...). Changes only: block names
(*_c30), keys ep 30-39, the label check is strict (z["mi_cell"], no fallback; mi_cell == mr_cell == path cell and
mi_block == path block for every file, else the BLOCK fails), new per-file key-set check (task 0-9, ep 30-39, init ==
RandomState(5478 + task).permutation(50)[ep], else the CELL fails). Never reads success.

Original docstring:
maskinfo gates (protocol_maskinfo.md section 4) -> gate json read by prereg_maskinfo.py. Never reads success.

Pre-submission part (no formal dumps needed):
  --manip FILES   manip_maskinfo.py snapshot-mode jsonl (2.4 items 1-10, 4 item 1 render determinism per card type,
                  4 item 5 physics)
  --t0 FILES      manip_maskinfo.py --t0 jsonl (4 item 3, ep 20-29 keys)
Formal part (after the runs; UNK until then):
  --runs ROOT     out/runs/<block>/<cell>/*.npz: per-episode hook self-checks, model-input md5, same sub-step,
                  copy_grip_cmd, scene (step-0 object qpos and reference fixtures vs the block's C0 and vs the oft
                  block of the same suite), card type per episode, physics breakdown detector D
  --logs GLOB     job logs: QACC lines per cell, OFTi missing-weight warnings, MI_CFG lines
Output: --json FILE (schema in prereg_maskinfo.py; per block card_type, block_ok, cells[c].status / bad_keys,
physics_breakdown[c]) and a short text summary on stdout (counts only, never outcomes / episode lengths).
"""
import argparse, glob, hashlib, importlib.util, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

CELLS = [f"C{i}" for i in range(8)]
BLOCKS = ["oft_spatial_c30", "ofti_spatial_c30", "oft_object_c30", "ofti_object_c30"]   # confirm §2 item 2
ALLOWED = ("h200", "h100-96")  # protocol v3 4 item 2 + deviations file (h100-96 only with its own render / t0 PASS)
EPS = list(range(30, 40))   # confirm §2 item 1
SHARDS = {30: range(30, 35), 35: range(35, 40)}
N_INIT = 50                 # confirm §2 item 1: scene = RandomState(5478 + task).permutation(50)[ep]


def expected_init(task, ep):
    return int(np.random.RandomState(5478 + task).permutation(N_INIT)[ep])


def card_type(gpu):
    g = str(gpu)
    if "H200" in g:
        return "h200"
    if "H100" in g:
        m = re.search(r"(\d+)\s*MiB", g)
        return "h100-96" if m and int(m.group(1)) >= 90000 else "h100-47"
    if "A100" in g and "80GB" in g:
        m = re.search(r"(\d+)\s*MiB", g)
        if m and int(m.group(1)) < 70000:
            return "a100-80-slice"  # node23 a100-40 slices report the 80GB name; memory tells them apart
        return "a100-80"
    return "other:" + g


def tol_ok(n, mx):
    return n == 0 or (n <= 10 and mx <= 1)


def jsonl(files):
    for f in files:
        for l in open(f):
            if l.strip():
                yield json.loads(l)


# ------------------------------------------------------------------ pre-submission
def summarize_manip(files):
    rows = list(jsonl(files))
    hdr = [r for r in rows if r.get("header")]
    keys = [r for r in rows if r.get("key_row") and not r.get("t0_mode")]
    fr = [r for r in rows if "view" in r and not r.get("t0_mode")]
    out = {"n_keys": len(keys), "n_frame_views": len(fr), "hosts_gpus": sorted({(r["host"], r["gpu"]) for r in keys})}
    # render determinism per card type (4 item 1)
    det = defaultdict(lambda: {"frames": 0, "seg_fail": 0, "rgb_fail": 0})
    gpu_of = {(r["task"], r["ep"]): card_type(r["gpu"]) for r in keys}
    for r in fr:
        ct = gpu_of.get((r["task"], r["ep"]), "unknown")
        for c, d in (r.get("det") or {}).items():
            if not d:
                continue
            x = det[ct]
            x["frames"] += 1
            x["seg_fail"] += int(d["seg_twice_px"] != 0 or d["bgseg_twice_px"] != 0)
            x["rgb_fail"] += int(not (tol_ok(*d["rgb_twice"]) and tol_ok(*d["bg_twice"])))
    out["render_det"] = {ct: dict(x, status="PASS" if x["frames"] > 0 and x["seg_fail"] == 0
                                  and x["rgb_fail"] <= 0.01 * x["frames"] else "FAIL") for ct, x in det.items()}
    # physics (4 item 5 / 2.4 item 10)
    ph = [max((v or 0.0) if v is not None else float("inf") for v in r["physics_qpos_max_vs_C0"].values()) for r in keys]
    out["physics"] = {"keys": len(ph), "max": max(ph) if ph else None,
                      "status": "PASS" if ph and max(ph) == 0.0 else "FAIL"}
    # item 1 (category check), 3, 4, 9, self checks
    bad = Counter()
    for r in fr:
        for c, d in r["item1"].items():
            if isinstance(d, dict) and "ok" in d and not d["ok"]:
                bad["item1:" + c] += 1
            if isinstance(d, dict) and d.get("drawn_all_black") is False:
                bad["item1:drawn_not_black:" + c] += 1
        if r["item1"].get("C6_union_eq_C2_union") is False:
            bad["item1:C6_union"] += 1
        for c, d in r.get("item3", {}).items():
            if d.get("outside_dilate8_px"):
                bad["item3:" + c] += 1
        for nm in ("item4_occ", "item4_occP"):  # obj_of_interest and P (protocol v2 2.4 item 4)
            occ = r.get(nm, {})
            for c in ("C3", "C4", "C5"):
                if occ.get(c) not in (None, 0.0):
                    bad[f"{nm}:{c}!=0"] += 1
            if nm == "item4_occP" and occ.get("C6") != occ.get("C1"):
                bad[f"{nm}:C6!=C1"] += 1
        for c, ok in r["item9"].items():
            if not ok and r["step"] > 0:
                bad["item9:" + c] += 1
        for c, s in r["self"].items():
            if s.get("out_E_diff_px"):
                bad["self_outside_E:" + c] += 1
            if s.get("same_substep") is False:
                bad["same_substep:" + c] += 1
            if s.get("final_eq_edited") is False:
                bad["final_eq_edited:" + c] += 1
        it2 = r.get("item2", {})
        if it2.get("B_eq_M7_plus_trunc") is False:
            bad["item2:B!=M7+trunc"] += 1
        if it2.get("M7_ge_08B") is False:  # protocol v3 2.4 item 2
            bad["item2:M7<0.8B"] += 1
        for c in ("C4_border", "C5_border"):  # protocol v3 2.4 item 3 (no line along the image edge)
            if r.get("item3", {}).get(c, {}).get("ok") is False:
                bad["item3:" + c] += 1
    out["violations"] = dict(bad)
    out["status"] = "PASS" if not bad and out["physics"]["status"] == "PASS" else "FAIL"
    return out


def summarize_t0(files):
    rows = []
    for f in files:  # tag every row with the suite of its file header
        su = None
        for r in jsonl([f]):
            if r.get("header"):
                su = r["suite"]
            r["_suite"] = su
            rows.append(r)
    keys = [r for r in rows if r.get("key_row") and r.get("t0_mode")]
    fr = [r for r in rows if "view" in r and r.get("t0_mode")]
    cells = {c: {"bad_keys": set(), "tolerated": 0} for c in CELLS}
    for r in fr:
        k = (r["_suite"], r["task"], r["ep"])
        for c, d in r["item1"].items():
            if not isinstance(d, dict) or "ok" not in d:
                continue
            cc = c.split("_")[0]
            if cc in cells:
                if not d["ok"]:
                    cells[cc]["bad_keys"].add(k)
                elif d.get("out_n", d.get("n", 0)):
                    cells[cc]["tolerated"] += 1
        for c, s in r["self"].items():
            if s.get("same_substep") is False or s.get("final_eq_edited") is False or s.get("out_E_diff_px"):
                cells[c]["bad_keys"].add(k)
    pres = Counter()
    for r in keys:
        for n, k in r["object_checks"].get("P_name_present", {}).items():
            pres[n] += k
    pf = [v for r in keys for v in r["object_checks"].get("P_full_nonempty", [])]
    obj = {"gripper_instance_ok": all(r["object_checks"]["gripper_instance"] == "PandaGripper0" for r in keys),
           "site_px_max": max((max(x for x in r["object_checks"]["site_px"] if x is not None) for r in keys), default=None),
           "T_full_nonempty_frac": float(np.mean([v for r in keys for v in r["object_checks"]["T_full_nonempty"]])) if keys else None,
           "P_full_nonempty_frac": float(np.mean(pf)) if pf else None,
           "P_names_never_seen": sorted(n for n, k in pres.items() if k == 0)}
    obj["status"] = "PASS" if (obj["gripper_instance_ok"] and obj["site_px_max"] == 0 and (obj["P_full_nonempty_frac"] or 0) >= 0.95
                               and not obj["P_names_never_seen"]) else "FAIL"
    return {"n_keys": len(keys), "keys": sorted({(r["task"], r["ep"]) for r in keys}),
            "cells": {c: {"bad_keys": sorted(v["bad_keys"]), "tolerated_frames": v["tolerated"]} for c, v in cells.items()},
            "object_checks": obj, "gpus": sorted({r["gpu"] for r in keys})}


# ------------------------------------------------------------------ formal runs
def load_detector():
    p = os.path.expanduser("$PROJ_ROOT/vla_zoo/tools/physics_breakdown_scan.py")
    spec = importlib.util.spec_from_file_location("pbscan", p)
    m = importlib.util.module_from_spec(spec)
    sys.argv_backup = sys.argv
    spec.loader.exec_module(m)
    return m, hashlib.sha256(open(p, "rb").read()).hexdigest()


def scan_block(root, block, ref_scene, det):
    res = {"cells": {}, "physics_breakdown": {}, "gpus": Counter(), "n": {}, "label_fail": [], "keyset_fail": {}}
    scenes = {}
    for c in CELLS:
        d = os.path.join(root, block, c)
        bad, reasons, keys = {}, Counter(), set()
        pb = []
        kfail = []
        for f in sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz"))):
            z = np.load(f, allow_pickle=True)
            k = (int(z["task"]), int(z["ep"]))
            # confirm §4 item 10: strict label fields, no fallback; any miss / mismatch fails the block
            try:
                lab_ok = str(z["mi_cell"]) == str(z["mr_cell"]) == c and str(z["mi_block"]) == block
            except KeyError as ex:
                lab_ok = False
                res["label_fail"].append(f"{c}/{os.path.basename(f)}: missing {ex}")
            else:
                if not lab_ok:
                    res["label_fail"].append(f"{c}/{os.path.basename(f)}: mi_cell / mr_cell / mi_block != path")
            # confirm §4 item 9: key set ep 30-39 (task 0-9) and init == permutation(50)[ep]; any miss fails the cell
            if not (0 <= k[0] <= 9 and k[1] in EPS):
                kfail.append(f"{os.path.basename(f)}: key t{k[0]} e{k[1]} outside task 0-9 x ep 30-39")
            elif int(z["init"]) != expected_init(*k):
                kfail.append(f"{os.path.basename(f)}: init {int(z['init'])} != permutation(50)[ep]")
            if k in keys:
                bad[k] = "duplicate key"
                continue
            keys.add(k)
            ct = card_type(z["gpu"])
            res["gpus"][ct] += 1
            why = []
            al = json.loads(str(z["align_json"]))
            if al.get("copy_grip_cmd") != 1:
                why.append("copy_grip_cmd")
            chk = json.loads(str(z["mr_checks"]))
            for row in chk:
                for v in ("agent", "wrist"):
                    x = row.get(v, {})
                    if x.get("same_substep") is False or x.get("final_eq_edited") is False:
                        why.append("hook_selfcheck")
                if row.get("model_input_is_edited") is False:
                    why.append("model_md5")
            for row in json.loads(str(z["mi_rows"])):
                for v, x in row.items():
                    if x.get("out_E_diff_px"):
                        why.append("outside_E")
            if not lab_ok:
                why.append("cell/block label")
            # scene: step-0 non-robot qpos and reference fixtures
            names = [str(n) for n in z["joint_names"]]
            q0 = np.asarray(z["qpos"][0])
            sc = (q0, np.asarray(z["ref_body_pos"]))
            scenes.setdefault(k, {})[c] = sc
            ref = ref_scene.get(k) if ref_scene is not None else None
            base = ref if ref is not None else scenes[k].get("C0")
            if base is not None and (base[0].shape != q0.shape or np.abs(base[0] - q0).max() > 0 or
                                     np.abs(base[1] - sc[1]).max() > 0):
                why.append("scene")
            # detector D
            ai = [names.index(f"robot0_joint{i}") for i in range(1, 8)]
            fi = names.index("gripper0_finger_joint1") if "gripper0_finger_joint1" in names else None
            A, B, _ = det.detect(np.asarray(z["qpos"]), ai, None, -0.5, 1.2)
            if A or B:
                pb.append(list(k))
            if why:
                bad[k] = sorted(set(why))
                for w in set(why):
                    reasons[w] += 1
        expect = {(t, e) for t in range(10) for e in EPS}
        missing = sorted(expect - keys)
        status = "UNK" if missing else ("PASS" if not bad else "PASS")  # key-level failures -> bad_keys
        if kfail:
            status = "FAIL"   # confirm §4 item 9
            reasons["key set"] += len(kfail)
            res["keyset_fail"][c] = kfail
        res["cells"][c] = {"status": status, "n": len(keys), "missing": len(missing),
                           "bad_keys": sorted([list(k) for k in bad]), "reasons": dict(reasons)}
        res["physics_breakdown"][c] = sorted(pb)
    return res, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--manip", nargs="*", default=[])
    ap.add_argument("--t0", nargs="*", default=[])
    ap.add_argument("--runs", default="")
    ap.add_argument("--logs", default="")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    out = {"protocol_sha256": hashlib.sha256(open(a.protocol, "rb").read()).hexdigest(),
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    out["manip"] = summarize_manip(a.manip) if a.manip else None
    out["t0"] = summarize_t0(a.t0) if a.t0 else None
    rd = (out["manip"] or {}).get("render_det", {})
    out["blocks"] = {}
    if a.runs:
        det, dsha = load_detector()
        out["detector_sha256"] = dsha
        logs = sorted(glob.glob(a.logs)) if a.logs else []
        ref_scenes = {}
        for b in BLOCKS:
            if not os.path.isdir(os.path.join(a.runs, b)):
                continue
            suite = b.split("_")[1]
            r, scenes = scan_block(a.runs, b, ref_scenes.get(suite), det)
            if b.startswith("oft_"):
                ref_scenes[suite] = {k: v.get("C0") for k, v in scenes.items() if v.get("C0") is not None}
            types = set(r["gpus"])
            ct = next(iter(types)) if len(types) == 1 else ("mixed" if types else None)
            fails = []
            if ct not in ALLOWED:
                fails.append(f"card type {ct} ({dict(r['gpus'])})")
            elif rd.get(ct, {}).get("status") != "PASS":
                fails.append(f"render determinism not passed on {ct}")
            if out["manip"] is None or out["manip"]["status"] != "PASS":
                fails.append("manipulation checks / physics not PASS")
            if r["label_fail"]:   # confirm §4 item 10
                fails.append(f"label fields: {len(r['label_fail'])} files ({r['label_fail'][:3]})")
            # t0 bad keys apply to every block of the suite (model independent)
            if out["t0"]:
                for c in CELLS:
                    tk = [k[1:] for k in out["t0"]["cells"][c]["bad_keys"] if k[0] == suite]
                    r["cells"][c]["bad_keys"] = sorted({tuple(x) for x in r["cells"][c]["bad_keys"]} | {tuple(x) for x in tk})
                    r["cells"][c]["bad_keys"] = [list(x) for x in r["cells"][c]["bad_keys"]]
            else:
                fails.append("t0 gate missing")
            # logs: QACC per cell, OFTi load, MI_CFG
            qacc, warn, cfg = Counter(), 0, Counter()
            for f in logs:
                m = re.search(r"mic_(\w+?)_(C\d)_e\d+_", os.path.basename(f))
                if not m or m.group(1) != b:
                    continue
                txt = open(f, errors="replace").read()
                qacc[m.group(2)] += len(re.findall(r"QACC", txt))
                warn += len(re.findall(r"(?i)weights.*(not initialized|newly initialized)|missing keys|unexpected keys", txt))
                cfg.update(re.findall(r"MI_CFG: .*?num_images_in_input=(\d) use_proprio=(\w+)", txt))
            if warn:
                fails.append(f"missing-weight warnings {warn}")
            if logs and any(k != ("2", "True") for k in cfg):
                fails.append(f"MI_CFG {dict(cfg)}")
            r["qacc_lines"] = dict(qacc)
            r["card_type"] = ct
            r["block_fail_reasons"] = fails
            r["block_ok"] = not fails and all(x["status"] == "PASS" for x in r["cells"].values())
            r["gpus"] = dict(r["gpus"])
            out["blocks"][b] = r
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: list(o) if isinstance(o, (set, tuple)) else str(o))
    m = out["manip"] or {}
    print(f"manip: {m.get('status')} keys {m.get('n_keys')} violations {m.get('violations')} physics {m.get('physics')}")
    print(f"render_det: {json.dumps(m.get('render_det'))}")
    if out["t0"]:
        print(f"t0: keys {out['t0']['n_keys']} " + " ".join(f"{c}:{len(v['bad_keys'])}" for c, v in out['t0']['cells'].items())
              + f" object {out['t0']['object_checks']}")
    for b, r in out["blocks"].items():
        print(f"{b}: card {r['card_type']} block_ok {r['block_ok']} fails {r['block_fail_reasons']} "
              + " ".join(f"{c}:{x['status']}/{x['n']}/bad{len(x['bad_keys'])}" for c, x in r["cells"].items()))


if __name__ == "__main__":
    main()
