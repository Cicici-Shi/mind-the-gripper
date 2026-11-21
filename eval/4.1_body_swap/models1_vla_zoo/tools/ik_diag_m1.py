"""models1 read-only twin-IK diagnostic, 2026-11-05.

Purpose: describe WHERE the twin-IK residual (twin Panda pad vs physical pad, `twin_pos_err_mm`) exceeds 1 mm in the
models1 cells that failed (OFTi object obj_g_ta_fc; OpenVLA spatial g_tfc / g_tw_fc / g_ta_fc) plus controls,
so the gate failures can be explained without touching any outcome.

Basis: Fable ruling `protocol_models1_gate_rulings_fable.md` (2026-11-05), section 3 item 4 (all bullets except the
last one, the optional twin_reach_diag reachability re-solve, which is done elsewhere), and section 4 item 4
(all-frame vs query-frame proportion on OFT npz).

Groups:  oft_obj libero_object  (7 cells, npz)      -> out/ik_diag_m1_oft_obj_libero_object.{txt,json}
         openvla libero_spatial (6 cells, jsonl)    -> out/ik_diag_m1_openvla_libero_spatial.{txt,json}
         pi0 libero_spatial     (g_tfc, g_ta_fc)    -> out/ik_diag_m1_pi0_libero_spatial.{txt,json}

OUTCOME FIREWALL (read-out boundary 2026-11-09 12:00 SGT):
- npz: the `success` array is never accessed (np.load is lazy; only named fields are read).
- jsonl: the `success` key is deleted right after json.loads, before the record is stored.
- Single exception required by the ruling (3.4, OVLA bullet): `_a0_vs_outcome_overlap()` re-reads `success` for the
  OVLA sham_cg / sham_twin keys inside one function and returns ONE integer = |{a0 differs} & {outcome differs}|.
  No key list, no per-cell outcome, no rate leaves that function.
- Episode length (= query count) is a proxy of the outcome: per-episode fractions use the fixed step cap as
  denominator (frac_of_cap), never the episode's own length; query-count differences are printed only for the
  zero-channel pairs GTw-G and GTa-GTfc (inputs identical for OVLA), not for T0-S (see run()).

Only complete shards are used (shard = offset o in {0,5,10,15}, keys (task 0..9, ep o..o+4), complete = all 50 keys
present and error-free for that cell). Joint limits are read from the robosuite Panda MJCF via mujoco (needs a
compute node; on the login node mujoco model building dies with Illegal instruction, so run via ik_diag_m1.sbatch).

Usage: python ik_diag_m1.py [--groups oft_obj,openvla,pi0] [--out $PROJ_ROOT/vla_zoo/out]
"""
import argparse, glob, hashlib, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
OFFS = (0, 5, 10, 15)
TRIALS = 5
GROUPS = {
    "oft_obj": ("libero_object", "npz", ["sham_cg", "sham_twin", "grpkp100", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"]),
    "openvla": ("libero_spatial", "jsonl", ["sham_cg", "sham_twin", "grpkp100", "g_tfc", "g_tw_fc", "g_ta_fc"]),
    "pi0": ("libero_spatial", "jsonl", ["g_tfc", "g_ta_fc"]),
}
TWIN = {"sham_twin", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"}
THR = (1, 2, 5, 10)
MAX_STEPS_OFT = {"libero_spatial": 220, "libero_object": 280}  # run_libero_eval max_steps per suite
OFT_STRIDE = 8          # OFT query frames = pe[::8] (zoo_gate_m1.py)
NEAR_LIMIT_DEG = 1.0


def shard_of(e):
    return max(o for o in OFFS if o <= e)


# ---------------------------------------------------------------- loaders (no outcome)
def load_npz(model, suite, cell):
    pre = "obj_" if suite == "libero_object" else ""
    out = {}
    for f in glob.glob(os.path.join(Z, "runs", model, pre + cell, "t*_e*_i*.npz")):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        try:
            z = np.load(f, allow_pickle=True)
            r = {"contact_steps": int(z["contact_arm_env_contact_steps"]) if "contact_arm_env_contact_steps" in z.files else None,
                 "first_contact": int(z["contact_first_contact"]) if "contact_first_contact" in z.files else None,
                 "stall_steps": int(z["contact_stall_steps"]) if "contact_stall_steps" in z.files else None}
            if "twin_pos_err_mm" in z.files:
                r["pe"] = np.asarray(z["twin_pos_err_mm"], float)
                r["re"] = np.asarray(z["twin_rot_err_deg"], float)
                r["qpos"] = np.asarray(z["qpos"], float)
                r["jn"] = [str(x) for x in z["joint_names"]]
                r["act"] = np.asarray(z["action"], float)
                r["eef"] = np.asarray(z["eef_pos"], float)
                r["max_steps"] = MAX_STEPS_OFT[suite]
        except Exception as ex:  # partially written file
            print(f"[skip unreadable {f}: {ex!r}]", file=sys.stderr)
            continue
        out[k] = r
    return out


def load_jsonl(model, suite, cell):
    pre = "obj_" if suite == "libero_object" else ""
    out = {}
    for f in glob.glob(os.path.join(Z, "runs", model, f"{pre}{cell}_e*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # line being written
                continue
            r.pop("success", None)  # outcome firewall
            if r.get("error") is not None:
                continue
            k = (int(r["task"]), int(r["episode"]))
            if k in out:
                continue
            c = r.get("contacts") or {}
            tw = r.get("twin")
            rec = {"contact_steps": c.get("arm_env_contact_steps"), "first_contact": c.get("first_contact"),
                   "stall_steps": c.get("stall_steps"), "queries": r.get("queries"), "a0": r.get("a0"),
                   "replan": r.get("replan"), "max_steps": r.get("max_steps"), "first_grip_obj": c.get("first_grip_obj")}
            if tw is not None:
                rec.update(pe_q=np.asarray(tw.get("pos_err_mm_q") or [], float), n_paint=tw.get("n_paint"),
                           over1_all=tw.get("over_1mm"), over1_q=tw.get("over_1mm_q"), miss=tw.get("miss"),
                           rot_max=tw.get("rot_err_deg_max"), pos_max_all=tw.get("pos_err_mm_max"),
                           has_rot_q="rot_err_deg_q" in tw)
            out[k] = rec
    return out


def complete(recs):
    have = Counter(shard_of(e) for (t, e) in recs)
    sh = [o for o in OFFS if have.get(o, 0) == 10 * TRIALS]
    return sh, {k: v for k, v in recs.items() if shard_of(k[1]) in sh}


# ---------------------------------------------------------------- helpers
def q(x, p):
    return round(float(np.percentile(x, p)), 4) if len(x) else None


def dist_stats(x):
    x = np.asarray(x, float)
    d = {"n": int(len(x)), "p50": q(x, 50), "p90": q(x, 90), "p99": q(x, 99), "max": round(float(x.max()), 4) if len(x) else None}
    for t in THR:
        d[f"gt{t}mm"] = int((x > t).sum())
    d["frac_gt1mm"] = round(float((x > 1).mean()), 5) if len(x) else None
    return d


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def min_eps_for(counts, share=0.8):
    tot = sum(counts)
    if tot == 0:
        return 0
    acc = 0
    for i, c in enumerate(sorted(counts, reverse=True), 1):
        acc += c
        if acc >= share * tot - 1e-9:
            return i
    return len(counts)


def stall_per_step(act, eef):
    """Re-derive embody/contacts.py stall rule per control step k (k >= 10, mean |a[:3]| over actions k-10..k-1 > 0.5
    and site moved < 5 mm between after-step k-10 and after-step k). after-step k position = eef[k+1]; the last step has
    no recorded after-position and is not evaluated. Returns boolean array over steps 0..len(act)-1."""
    n = len(act)
    s = np.zeros(n, bool)
    for k in range(10, n - 1):
        if np.linalg.norm(act[k - 10:k, :3], axis=1).mean() > 0.5 and np.linalg.norm(eef[k + 1] - eef[k - 9]) < 0.005:
            s[k] = True
    return s


def panda_limits():
    """robosuite Panda jnt_range for joint1..7, read through mujoco (compute node only). The Panda robot.xml alone has no
    <compiler angle="radian"> (it comes from robosuite's base.xml), so the model is compiled the way robosuite does it:
    MujocoWorldBase (base.xml) + merge(Panda()). Sanity check: joint4 upper limit < 0 and joint1 span > 5 rad."""
    try:
        import mujoco, robosuite
        from robosuite.models import MujocoWorldBase
        from robosuite.models.robots import Panda
        w = MujocoWorldBase()
        w.merge(Panda())
        m = mujoco.MjModel.from_xml_string(w.get_xml())
        lim = {}
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            mm = re.fullmatch(r"(?:robot\d+_)?joint([1-7])", nm or "")
            if mm:
                lim[int(mm.group(1))] = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1]), bool(m.jnt_limited[j]))
        assert sorted(lim) == list(range(1, 8)), lim
        assert lim[4][1] < 0 and lim[1][1] - lim[1][0] > 5, f"implausible Panda limits (degree/radian?) {lim}"
        xml = os.path.join(os.path.dirname(robosuite.__file__), "models", "assets", "robots", "panda", "robot.xml")
        return {"source": "robosuite MujocoWorldBase + Panda()", "robosuite": getattr(robosuite, "__version__", "?"),
                "robot_xml_sha256": hashlib.sha256(open(xml, "rb").read()).hexdigest(), "range_rad": lim}
    except Exception as ex:
        return {"error": repr(ex)}


# ---------------------------------------------------------------- per-cell IK diagnostic
def ik_cell(kind, recs, lims):
    R = {}
    keys = sorted(k for k, r in recs.items() if ("pe" in r or "pe_q" in r))
    if not keys:
        return {"twin_records": 0}
    per_ep = []
    task_over = defaultdict(int); task_eps = defaultdict(int)
    allq, allf, rot_q, rot_all = [], [], [], []
    near = Counter(); near_frames = 0; frames_with_q = 0; len_mismatch = 0
    near_clean = Counter(); clean_frames = 0; clean_near = 0; mind_over, mind_clean, jmin_over = [], [], []
    stall_ov = {"over_frames": 0, "over_frames_in_stall": 0, "stall_steps_rederived": 0, "stall_steps_recorded": 0}
    for k in keys:
        r = recs[k]
        if kind == "npz":
            pe, re_ = r["pe"], r["re"]
            pq = pe[::OFT_STRIDE]
            allf.extend(pe.tolist()); rot_all.extend(re_.tolist()); rot_q.extend(re_[::OFT_STRIDE].tolist())
            ep_series = pe  # per-episode stats on all frames (also give query count)
            cap = int(r["max_steps"]) + 1
            nq_over = int((pq > 1).sum())
            if len(pe) != len(r["act"]) + 1:
                len_mismatch += 1
            over_idx = np.where(pe > 1)[0]
            # joint-limit proximity of the physical arm at over-line frames (pe[k] <-> qpos[k], same instant)
            if lims and "range_rad" in lims:
                jn = r["jn"]; idx = [jn.index(f"robot0_joint{i}") for i in range(1, 8)]
                assert idx == list(range(7)), "robot joints are not qpos[0:7]"  # hinge joints first in sim qpos
                lo = np.array([lims["range_rad"][i][0] for i in range(1, 8)]); hi = np.array([lims["range_rad"][i][1] for i in range(1, 8)])
                lm = np.array([lims["range_rad"][i][2] for i in range(1, 8)])
                Q = r["qpos"][:, :7]
                dd = np.degrees(np.minimum(Q - lo, hi - Q)); dd[:, ~lm] = np.inf   # (steps, 7) distance to nearest limit, deg
                n = min(len(pe), len(Q))
                ov = (pe[:n] > 1)
                frames_with_q += int(ov.sum())
                nearm = dd[:n] < NEAR_LIMIT_DEG
                near_frames += int(nearm[ov].any(1).sum())
                for i in range(7):
                    near[i + 1] += int(nearm[ov, i].sum()); near_clean[i + 1] += int(nearm[~ov, i].sum())
                clean_frames += int((~ov).sum()); clean_near += int(nearm[~ov].any(1).sum())
                mind_over.extend(dd[:n][ov].min(1).tolist()); mind_clean.extend(dd[:n][~ov].min(1).tolist())
                jmin_over.extend(np.argmin(dd[:n][ov], 1).tolist())
            st = stall_per_step(r["act"], r["eef"])
            stall_ov["stall_steps_rederived"] += int(st.sum()); stall_ov["stall_steps_recorded"] += int(r["stall_steps"] or 0)
            stall_frames = set((np.where(st)[0] + 1).tolist())  # stall at step k -> frame after step k = pe[k+1]
            stall_ov["over_frames"] += int(over_idx.size)
            stall_ov["over_frames_in_stall"] += sum(1 for fi in over_idx if int(fi) in stall_frames)
        else:
            pq = r["pe_q"]
            ep_series = pq
            cap = int(r["max_steps"]) // int(r.get("replan") or 1) + 1
            nq_over = int((pq > 1).sum())
        allq.extend(pq.tolist())
        mask = ep_series > 1
        n_over = int(mask.sum())
        first = int(np.argmax(mask)) if n_over else None
        if kind == "jsonl" and first is not None:
            first = first * int(r.get("replan") or 1)  # query index -> step
        e = {"key": list(k), "over1": n_over, "frac_of_cap": round(n_over / cap, 4),
             "longest_run": longest_run(mask), "first_over_step": first,
             "max_mm": round(float(ep_series.max()), 3) if len(ep_series) else None,
             "contact": int((r["contact_steps"] or 0) > 0), "first_contact": r["first_contact"],
             "stall": int((r["stall_steps"] or 0) > 0)}
        if kind == "npz":
            e["over1_q"] = nq_over
        per_ep.append(e)
        task_over[k[0]] += n_over; task_eps[k[0]] += int(n_over > 0)
    frame_basis = "all frames" if kind == "npz" else "query frames"
    R["query_frames"] = dist_stats(allq)
    if kind == "npz":
        R["all_frames"] = dist_stats(allf)
        R["query_vs_all_frac_gt1mm"] = {"query": R["query_frames"]["frac_gt1mm"], "all": R["all_frames"]["frac_gt1mm"],
                                        "query_gt2": R["query_frames"]["gt2mm"], "all_gt2": R["all_frames"]["gt2mm"]}
        # offset sensitivity: every phase of the stride, to show whether pe[::8] is a typical phase
        R["query_phase_frac_gt1mm"] = None
        for nm, x in (("rot_deg_query", rot_q), ("rot_deg_all", rot_all)):
            x = np.asarray(x, float)
            R[nm] = {"n": int(len(x)), "p50": q(x, 50), "p90": q(x, 90), "p99": q(x, 99), "max": round(float(x.max()), 4),
                     "gt1deg": int((x > 1).sum()), "gt5deg": int((x > 5).sum())}
        R["retries"] = "no field (BodyEnv twin_stats['retries'] is not dumped to npz)"
        R["episode_len_mismatch(pe != action+1)"] = len_mismatch
        R["joint_limits"] = ({"basis": "physical arm qpos[k] at frame pe[k] (same instant), all frames",
                              "over1_frames": frames_with_q, "over1_frames_any_joint_within_1deg": near_frames,
                              "over1_per_joint_within_1deg": dict(sorted(near.items())),
                              "clean_frames": clean_frames, "clean_frames_any_joint_within_1deg": clean_near,
                              "clean_per_joint_within_1deg": dict(sorted(near_clean.items())),
                              "min_dist_to_limit_deg_over1": {"p10": q(mind_over, 10), "p50": q(mind_over, 50), "p90": q(mind_over, 90)},
                              "min_dist_to_limit_deg_clean": {"p10": q(mind_clean, 10), "p50": q(mind_clean, 50), "p90": q(mind_clean, 90)},
                              "nearest_joint_over1_count": dict(sorted(Counter(int(j) + 1 for j in jmin_over).items()))}
                             if lims and "range_rad" in lims else "limits unavailable")
        R["stall_overlap_frame_level"] = stall_ov
    else:
        R["rot_deg"] = ("no per-frame field (jsonl twin has only rot_err_deg_max per episode and miss = #frames with pos>2mm or rot>1deg)"
                        if not any(recs[k].get("has_rot_q") for k in keys) else "per-frame field present")
        rm = [recs[k]["rot_max"] for k in keys if recs[k].get("rot_max") is not None]
        R["rot_deg_episode_max"] = {"p50": q(rm, 50), "p90": q(rm, 90), "max": round(float(max(rm)), 4) if rm else None,
                                    "eps_gt1deg": sum(1 for x in rm if x > 1)}
        R["miss_all_frames(pos>2|rot>1)"] = int(sum(recs[k].get("miss") or 0 for k in keys))
        o1a = sum(recs[k].get("over1_all") or 0 for k in keys); npaint = sum(recs[k].get("n_paint") or 0 for k in keys)
        o1q = sum(recs[k].get("over1_q") or 0 for k in keys)
        R["all_vs_query_frac_gt1mm"] = {"all": round(o1a / max(npaint, 1), 5), "query": R["query_frames"]["frac_gt1mm"],
                                        "query_from_client_counter": round(o1q / max(len(allq), 1), 5)}
        R["retries"] = "no field (zoo_client does not record twin_stats['retries'])"
        R["joint_limits"] = "no per-step qpos in jsonl"
    R["episodes"] = len(per_ep)
    R["per_episode_basis"] = frame_basis
    eps_over = [e for e in per_ep if e["over1"] > 0]
    R["eps_with_gt1mm"] = len(eps_over)
    R["min_eps_for_80pct_of_over_frames"] = min_eps_for([e["over1"] for e in per_ep])
    R["top_episodes"] = sorted(eps_over, key=lambda e: -e["over1"])[:10]
    R["by_task"] = {t: {"over_frames": task_over[t], "eps_with_over": task_eps[t]} for t in sorted(task_over)}
    # episode-level overlap with contact / stall
    def rate(lst, f):
        return f"{sum(e[f] for e in lst)}/{len(lst)}"
    clean = [e for e in per_ep if e["over1"] == 0]
    R["episode_overlap"] = {"over_eps_with_arm_env_contact": rate(eps_over, "contact"), "clean_eps_with_arm_env_contact": rate(clean, "contact"),
                            "over_eps_with_stall": rate(eps_over, "stall"), "clean_eps_with_stall": rate(clean, "stall"),
                            "over_eps_first_over_after_first_contact": sum(1 for e in eps_over if e["first_contact"] is not None and e["first_contact"] >= 0
                                                                           and e["first_over_step"] is not None and e["first_over_step"] >= e["first_contact"])}
    over_frames_total = sum(e["over1"] for e in per_ep)
    R["over_frames_in_contact_eps"] = f"{sum(e['over1'] for e in eps_over if e['contact'])}/{over_frames_total}"
    R["over_frames_in_stall_eps"] = f"{sum(e['over1'] for e in eps_over if e['stall'])}/{over_frames_total}"
    return R


def phase_check(recs):
    """OFT: fraction > 1 mm for each of the 8 phases of the stride (phase 0 = query frames)."""
    out = {}
    for ph in range(OFT_STRIDE):
        x = np.concatenate([recs[k]["pe"][ph::OFT_STRIDE] for k in recs if "pe" in recs[k]])
        out[ph] = round(float((x > 1).mean()), 5)
    return out


# ---------------------------------------------------------------- OVLA paired interface comparisons
def paired(a, b, A, B, fields=("first_contact", "first_grip_obj", "stall_steps", "queries")):
    ks = sorted(set(A) & set(B))
    res = {"pair": f"{a} - {b}", "keys": len(ks)}
    for f in fields:
        d = [A[k][f] - B[k][f] for k in ks if A[k].get(f) is not None and B[k].get(f) is not None]
        d = np.asarray(d, float)
        res[f] = {"n": int(len(d)), "eq0": int((d == 0).sum()), "abs_p50": q(np.abs(d), 50), "abs_p90": q(np.abs(d), 90),
                  "abs_max": float(np.abs(d).max()) if len(d) else None, "mean": round(float(d.mean()), 2) if len(d) else None}
    res["a0_bitwise_identical"] = sum(1 for k in ks if A[k].get("a0") is not None and A[k]["a0"] == B[k]["a0"])
    return res


def _a0_vs_outcome_overlap(model, suite, ka, kb, keys, A, B):
    """THE ONLY place that touches `success` (ruling 3.4, OVLA bullet). Returns one integer:
    #keys whose first executed action differs AND whose outcome differs. Nothing else escapes."""
    def outc(cell):
        pre = "obj_" if suite == "libero_object" else ""
        o = {}
        for f in glob.glob(os.path.join(Z, "runs", model, f"{pre}{cell}_e*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    o.setdefault((int(r["task"]), int(r["episode"])), bool(r["success"]))
        return o
    oa, ob = outc(ka), outc(kb)
    n = sum(1 for k in keys if A[k].get("a0") != B[k].get("a0") and oa.get(k) != ob.get(k))
    del oa, ob
    return int(n)


# ---------------------------------------------------------------- main
def run(group, outdir, lims):
    suite, kind, cells = GROUPS[group]
    loader = load_npz if kind == "npz" else load_jsonl
    lines, J = [], {"group": group, "model": group, "suite": suite, "ruling": "protocol_models1_gate_rulings_fable.md 3.4 / 4.4",
                    "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "cells": {}}
    P = lines.append
    P(f"# ik_diag_m1  {group}  {suite}   (no outcome field read; see script header)")
    P(f"script sha256 {J['script_sha256']}")
    if kind == "npz":
        P(f"panda limits: {json.dumps(lims)}")
        J["panda_limits"] = lims
    data = {}
    for c in cells:
        recs = loader(group, suite, c)
        sh, rc = complete(recs)
        data[c] = rc
        J["cells"][c] = {"records_on_disk": len(recs), "complete_shards": sh, "keys_used": len(rc)}
        P(f"\n== {c}: complete shards {sh} -> {len(rc)} keys (records on disk {len(recs)})")
        if c in TWIN:
            R = ik_cell(kind, rc, lims if kind == "npz" else None)
            if kind == "npz" and rc:
                R["query_phase_frac_gt1mm"] = phase_check(rc)
            J["cells"][c].update(R)
            for k, v in R.items():
                if k == "top_episodes":
                    P("  top_episodes (key, over1, frac_of_cap, longest_run, first_over_step, max_mm, contact, first_contact, stall"
                      + (", over1_q" if kind == "npz" else "") + "):")
                    for e in v:
                        P("    " + json.dumps(e))
                else:
                    P(f"  {k}: {json.dumps(v, default=str)}")
    if group == "openvla":
        J["paired"] = {}
        # queries (= episode length, an outcome proxy) is compared only for the zero-channel pairs whose inputs are
        # meant to be identical (GTw vs G, GTa vs GTfc: OVLA reads the third-person view only). For T0 vs S the
        # trajectories diverge by design, and the number of equal-length keys would bound the count of keys where
        # both episodes ran to the step cap, i.e. leak outcome information; queries is therefore omitted there.
        NOQ = ("first_contact", "first_grip_obj", "stall_steps")
        for a, b, flds in (("g_tw_fc", "grpkp100", None), ("g_ta_fc", "g_tfc", None), ("sham_twin", "sham_cg", NOQ)):
            if a in data and b in data:
                pr = paired(a, b, data[a], data[b]) if flds is None else paired(a, b, data[a], data[b], flds)
                J["paired"][f"{a}-{b}"] = pr
                P(f"\n== paired {json.dumps(pr)}")
        if "sham_twin" in data and "sham_cg" in data:
            ks = sorted(set(data["sham_twin"]) & set(data["sham_cg"]))
            n = _a0_vs_outcome_overlap(group, suite, "sham_twin", "sham_cg", ks, data["sham_twin"], data["sham_cg"])
            J["T0_vs_S_a0diff_and_outcomediff_overlap"] = {"keys": len(ks), "a0_differs": sum(1 for k in ks if data["sham_twin"][k].get("a0") != data["sham_cg"][k].get("a0")), "overlap": n}
            P(f"\n== T0 vs S: keys {len(ks)}, a0 differs {J['T0_vs_S_a0diff_and_outcomediff_overlap']['a0_differs']}, "
              f"overlap(a0 differs AND outcome differs) = {n}")
    base = os.path.join(outdir, f"ik_diag_m1_{group}_{suite}")
    open(base + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(J, open(base + ".json", "w"), indent=1, default=str)
    for ext in (".txt", ".json"):
        print(f"{base}{ext} sha256 {hashlib.sha256(open(base + ext, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="oft_obj,openvla,pi0")
    ap.add_argument("--out", default=os.path.join(Z, "out"))
    a = ap.parse_args()
    gs = a.groups.split(",")
    lims = panda_limits() if any(GROUPS[g][1] == "npz" for g in gs) else None  # mujoco only needed for npz groups

    for g in gs:
        run(g, a.out, lims)
