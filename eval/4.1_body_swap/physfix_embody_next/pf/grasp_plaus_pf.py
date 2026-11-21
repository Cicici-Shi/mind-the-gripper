"""physfix rev3 7.1 grasp-physics plausibility (Astra weak point 3; impl list item 4), 2026-11-08. CPU, physics only,
reads npz qpos / action / pf_grip_contact, NEVER `success`; writes no episode length, no step index, no per-key step
count (per-key files hold only derived ratios / magnitudes). Definitions fixed here before any pf_ npz was processed.

Method: state reconstruction, not replay. For each key a BodyEnv(PandaR85) of the same task is built (embody_next,
switches off; Robotiq geometry is the same for kp20 / kp100, only actuator gains differ, which do not enter collision
geometry), set_init_state(init), then for every dumped step t: data.qpos <- qpos[t], qvel 0, mj_forward, read contacts.
Contacts therefore are those of the recorded state exactly (penetration = three-dimensional contact dist, rev3 6.10);
contact FORCES are not recoverable this way (no qvel / constraint history) -> force only from the online log
(pf_grip_contact, kp20 cells), NA for the old column. The online log is also compared with the reconstruction
(gripper-vs-non-robot contact count and min dist per step) as a validity check; it is printed, not a gate.
Columns: kp20 = G'' (pf_obj_grpkp20) and GTfc'' (pf_obj_g_tfc_kp20) of each checkpoint, pooled and per cell;
old_normal = published third-batch obj_g (OFT combined, kp100, no armature) minus its detector-D keys
(out/physics_breakdown_published.json, oft_pub|libero_object|obj_g); the same old column serves both checkpoints.
Per key, FIRST grip segment: start = first step with action[-1] > 0 (close); end = first later step with action[-1] <= 0,
or the last step. Target = env.obj_of_interest[0] (its body subtree geoms). Pads = gripper0_{left,right}_fingerpad_collision.
  lift-off t_l   = first step in the segment with target root z >= z(start) + 0.01 m; lifted = t_l exists
  (a) dual_pad_frac = fraction of steps in [t_l, end] (or [start, end] if not lifted) with both pads touching the target;
      pad contact in-face fraction = share of pad-target contacts whose point lies inside the pad box face (the two
      box axes other than the one most aligned with the hand opening axis, 0.5 mm margin)
  (b) penetration_mm = -(min dist over gripper-target contacts in [start, end]) * 1000 (positive = penetration);
      segments without any gripper-target contact excluded
  (c) slip_mm = p90 over [t_l, end] of |r(t) - r(t_l)|, r = target root position in the hand frame relative to the grip site
  (d) force_N = median over [start, end] of the online summed normal force (online log, all gripper-vs-non-robot contacts)
  (e) carry_frac = fraction of steps in [t_l, end] with | |p_obj - p_site|(t) - |p_obj - p_site|(t_l) | < 0.01 m;
      carried = lifted and carry_frac >= 0.5
  (f) passive_over: per key over the whole episode, the 5 passive Robotiq joints vs their model jnt_range: any step
      beyond (bool) and the largest excess (rad); no step count (length proxy)
  no_pad_carried = carried and no pad-target contact at any step of [t_l, end]
Cell / column summaries (keys with a segment only): no_pad_contact_carried_frac = #no_pad_carried / #segments;
penetration_p50/p90/max; carry_frac_median (lifted segments); dual_pad_frac_median; slip_p50/p90; force_median;
passive keys_over / max_excess; plus n_segments (n_lifted / n_carried withheld: success proxy).
Hand-off triggers (rev3 7.1, numbers by Fable; only decide "hand to Astra"): kp20 no_pad_contact_carried_frac > 0.01, or
penetration_p90 > 2 x old_normal, or carry_frac_median < 0.8 x old_normal. Evaluated on each checkpoint's pooled kp20
column AND on each kp20 cell; any trigger anywhere -> TRIGGER (conservative, fixed before the run). Slice 0 (ep 0-4) and
full (ep 0-19) both summarised.
Usage: --mode key --cell DIR --task T --out DIR   (all keys of one task of one cell)
       --mode summ --indir DIR --out grasp_physfix.json
"""
import argparse, glob, json, os, re, sys
import numpy as np

H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["key", "summ"])
ap.add_argument("--cell"); ap.add_argument("--task", type=int); ap.add_argument("--tag")
ap.add_argument("--indir"); ap.add_argument("--out", required=True)
ap.add_argument("--code", default=H("$PROJ_ROOT/embody_next"))
ap.add_argument("--breakdown", default=H("$PROJ_ROOT/vla_zoo/out/physics_breakdown_published.json"))
a = ap.parse_args()
PASSIVE = ["left_inner_finger_joint", "left_inner_knuckle_joint", "right_outer_knuckle_joint", "right_inner_finger_joint",
           "right_inner_knuckle_joint"]


def key_mode():
    for k in ("R85_ARMATURE", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "BODY"):
        os.environ.pop(k, None)
    os.environ.update(COPY_GRIP_CMD="1", ORIGINAL_ONLY="1")
    sys.path.insert(0, a.code)
    import mujoco  # noqa
    import embody  # noqa
    assert os.path.abspath(embody.__file__).startswith(os.path.abspath(a.code))
    from embody.body_env import BodyEnv  # noqa
    from embody.original import OriginalObjectSuite  # noqa
    from libero.libero import get_libero_path  # noqa
    from libero.libero.envs.env_wrapper import ControlEnv  # noqa
    suite = OriginalObjectSuite()
    task = suite.get_task(a.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    np.random.seed(7)
    ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    be = BodyEnv(ref, bddl, "PandaR85", 256, 10, log=lambda m_: None, render=False)
    be.reset()
    inits = suite.get_task_init_states(a.task)
    sim = be.env.sim
    m, d = sim.model, sim.data
    mr, dr = m._model, d._data
    rb = be.env.robots[0]
    e = be.env.env
    oname = e.obj_of_interest[0]
    obj = e.objects_dict[oname]
    oroot = m.body_name2id(obj.root_body)

    def in_sub(b):
        while b > 0:
            if b == oroot:
                return True
            b = m.body_parentid[b]
        return False
    OG = {g for g in range(m.ngeom) if in_sub(m.geom_bodyid[g])}
    GR = {g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")}
    RB = {g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(("robot0", "gripper0", "mount0"))}
    PADS = {m.geom_name2id("gripper0_left_fingerpad_collision"): "L", m.geom_name2id("gripper0_right_fingerpad_collision"): "R"}
    hb = m.body_name2id(rb.robot_model.eef_name)
    sid = rb.eef_site_id
    jn = [m.joint_id2name(j) for j in range(m.njnt)]
    pj = [m.joint_name2id("gripper0_" + j) for j in PASSIVE]
    pqa = [m.jnt_qposadr[j] for j in pj]
    prng = [tuple(m.jnt_range[j]) if m.jnt_limited[j] else None for j in pj]
    out = []
    for f in sorted(glob.glob(os.path.join(a.cell, f"t{a.task}_e*_i*.npz"))):
        mm = re.search(r"t(\d+)_e(\d+)_i(\d+)\.npz$", f)
        t_, ep, ini = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        if ep >= 20:
            continue
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["joint_names"]]
        assert names == jn, ("joint order differs", f)
        Q = np.asarray(z["qpos"], float)
        assert Q.shape[1] == m.nq, f
        A = np.asarray(z["action"], float)
        live = np.asarray(z["pf_grip_contact"], float) if "pf_grip_contact" in z.files else None
        be.set_init_state(inits[ini])
        oqa = m.get_joint_qpos_addr(obj.joints[0])[0]
        rec = {"task": t_, "ep": ep, "init": ini, "obj": oname}
        # (f) passive joints, whole episode
        exc = 0.0
        for a_, r_ in zip(pqa, prng):
            if r_ is None:
                continue
            q = Q[:, a_]
            exc = max(exc, float(np.max(np.maximum(r_[0] - q, q - r_[1]))))
        rec["passive_over"] = bool(exc > 0); rec["passive_max_excess_rad"] = round(max(exc, 0.0), 4)
        cl = np.nonzero(A[:, -1] > 0)[0]
        if not len(cl):
            rec["segment"] = False
            out.append(rec); continue
        s0 = int(cl[0])
        op = np.nonzero(A[s0:, -1] <= 0)[0]
        s1 = s0 + int(op[0]) if len(op) else len(A) - 1
        n = min(s1 + 1, len(Q))
        rows = []
        for t in range(s0, n):
            d.qpos[:] = Q[t]; d.qvel[:] = 0.0
            mujoco.mj_forward(mr, dr)
            u = d.xmat[hb].reshape(3, 3)[:, 1]
            pad_t, mind_t, ng, gmin, inface = set(), np.nan, 0, np.nan, []
            for i in range(dr.ncon):
                c = dr.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in GR) != (g2 in GR):
                    mine, oth = (g1, g2) if g1 in GR else (g2, g1)
                    if oth not in RB:
                        ng += 1; gmin = c.dist if np.isnan(gmin) else min(gmin, c.dist)
                    if oth in OG:
                        mind_t = c.dist if np.isnan(mind_t) else min(mind_t, c.dist)
                        if mine in PADS:
                            pad_t.add(PADS[mine])
                            R = d.geom_xmat[mine].reshape(3, 3); sz = m.geom_size[mine]
                            loc = R.T @ (np.asarray(c.pos) - d.geom_xpos[mine])
                            nax = int(np.argmax(np.abs(R.T @ u)))
                            inface.append(all(abs(loc[j]) <= sz[j] + 5e-4 for j in range(3) if j != nax))
            Rh = d.xmat[hb].reshape(3, 3)
            po, ps = d.qpos[oqa:oqa + 3].copy(), d.site_xpos[sid].copy()
            rows.append({"pad": pad_t, "mind": mind_t, "po": po, "ps": ps, "r": Rh.T @ (po - ps), "ng": ng, "gmin": gmin,
                         "inface": inface, "t": t})
        z0 = rows[0]["po"][2]
        tl = next((i for i, r in enumerate(rows) if r["po"][2] >= z0 + 0.01), None)
        W = rows[tl:] if tl is not None else rows
        rec["segment"] = True
        rec["lifted"] = tl is not None
        rec["dual_pad_frac"] = float(np.mean([len(r["pad"]) == 2 for r in W]))
        rec["any_pad_frac"] = float(np.mean([len(r["pad"]) > 0 for r in W]))
        inf = [x for r in rows for x in r["inface"]]
        rec["pad_inface_frac"] = float(np.mean(inf)) if inf else None
        md = [r["mind"] for r in rows if not np.isnan(r["mind"])]
        rec["penetration_mm"] = float(-min(md) * 1000) if md else None
        if tl is not None:
            d0 = np.linalg.norm(W[0]["po"] - W[0]["ps"])
            cf = float(np.mean([abs(np.linalg.norm(r["po"] - r["ps"]) - d0) < 0.01 for r in W]))
            rec["carry_frac"] = cf
            rec["carried"] = cf >= 0.5
            rec["slip_mm"] = float(np.percentile([np.linalg.norm(r["r"] - W[0]["r"]) for r in W], 90) * 1000)
            rec["no_pad_carried"] = bool(rec["carried"] and not any(len(r["pad"]) for r in W))
        else:
            rec["carry_frac"] = None; rec["carried"] = False; rec["slip_mm"] = None; rec["no_pad_carried"] = False
        if live is not None and len(live) >= n:
            L = live[s0:n]
            rec["force_N_median"] = float(np.median(L[:, 2]))
            # validity: online (end of control step) vs reconstruction (dumped qpos) -- same step index assumed
            ln, lmin = L[:, 0], L[:, 1]
            rn = np.array([r["ng"] for r in rows]); rmin = np.array([r["gmin"] for r in rows])
            rec["live_vs_recon_count_equal_frac"] = float(np.mean(ln == rn))
            both = ~np.isnan(lmin) & ~np.isnan(rmin)
            rec["live_vs_recon_mindist_absdiff_mm_p90"] = float(np.percentile(np.abs(lmin[both] - rmin[both]) * 1000, 90)) if both.any() else None
        out.append(rec)
    be.close()
    json.dump(out, open(a.out, "w"))
    print("done", a.out, len(out))


def summ_col(R):
    seg = [r for r in R if r.get("segment")]
    lif = [r for r in seg if r.get("lifted")]
    pen = [r["penetration_mm"] for r in seg if r.get("penetration_mm") is not None]
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    # n_lifted / n_carried are NOT output (2026-11-08, after run <job> printed them): on object tasks the number of
    # episodes that lift / carry the target is a strong proxy of the success rate; only ratios and magnitudes are kept
    o = {"n_keys": len(R), "n_segments": len(seg),
         "no_pad_contact_carried_n": sum(r["no_pad_carried"] for r in seg),
         "no_pad_contact_carried_frac": (sum(r["no_pad_carried"] for r in seg) / len(seg)) if seg else None,
         "penetration_p50": q(pen, 50), "penetration_p90": q(pen, 90), "penetration_max": max(pen) if pen else None,
         "carry_frac_median": q([r["carry_frac"] for r in lif], 50),
         "dual_pad_frac_median": q([r["dual_pad_frac"] for r in seg], 50),
         "any_pad_frac_median": q([r["any_pad_frac"] for r in seg], 50),
         "pad_inface_frac_median": q([r["pad_inface_frac"] for r in seg if r.get("pad_inface_frac") is not None], 50),
         "slip_p50": q([r["slip_mm"] for r in lif], 50), "slip_p90": q([r["slip_mm"] for r in lif], 90),
         "force_median": q([r["force_N_median"] for r in seg if r.get("force_N_median") is not None], 50),
         "passive_keys_over": sum(r["passive_over"] for r in R),
         "passive_max_excess_rad": max([r["passive_max_excess_rad"] for r in R] or [0.0])}
    lv = [r["live_vs_recon_count_equal_frac"] for r in seg if r.get("live_vs_recon_count_equal_frac") is not None]
    if lv:
        o["check_live_vs_recon_count_equal_frac_median"] = q(lv, 50)
        o["check_live_vs_recon_count_equal_frac_min"] = min(lv)
        lm = [r["live_vs_recon_mindist_absdiff_mm_p90"] for r in seg if r.get("live_vs_recon_mindist_absdiff_mm_p90") is not None]
        o["check_live_vs_recon_mindist_absdiff_mm_p90_median"] = q(lm, 50)
    return o


def triggers(k, old):
    t = []
    if k["no_pad_contact_carried_frac"] is not None and k["no_pad_contact_carried_frac"] > 0.01:
        t.append("no_pad_contact_carried_frac > 0.01")
    if k["penetration_p90"] is not None and old["penetration_p90"] is not None and k["penetration_p90"] > 2 * old["penetration_p90"]:
        t.append("penetration_p90 > 2 x old_normal")
    if k["carry_frac_median"] is not None and old["carry_frac_median"] is not None and k["carry_frac_median"] < 0.8 * old["carry_frac_median"]:
        t.append("carry_frac_median < 0.8 x old_normal")
    miss = [x for x in ("no_pad_contact_carried_frac", "penetration_p90", "carry_frac_median") if k[x] is None or old.get(x) is None]
    return t, miss


def summ_mode():
    J = {}
    for f in glob.glob(os.path.join(a.indir, "*.json")):
        tag = re.sub(r"_t\d+\.json$", "", os.path.basename(f))
        J.setdefault(tag, []).extend(json.load(open(f)))
    bd = json.load(open(a.breakdown))["_by_cell"]["oft_pub|libero_object|obj_g"]["breakdown"]
    bad = {tuple(x) for x in bd}
    res = {"definitions": __doc__, "old_normal_excluded_D_keys": sorted(bad), "slices": {}}
    anytrig = False
    for sl, emax in (("slice0", 5), ("full", 20)):
        S = {}
        old = summ_col([r for r in J["old_obj_g"] if r["ep"] < emax and (r["task"], r["ep"]) not in bad])
        S["old_normal"] = old
        for ck in ("OFT", "OFTi"):
            G = {"old_normal": old}
            cells = {c: [r for r in J[f"{ck}_{c}"] if r["ep"] < emax] for c in ("G2", "GTfc2")}
            G["kp20"] = summ_col(cells["G2"] + cells["GTfc2"])
            G["kp20_G2"] = summ_col(cells["G2"]); G["kp20_GTfc2"] = summ_col(cells["GTfc2"])
            tr = {}
            for col in ("kp20", "kp20_G2", "kp20_GTfc2"):
                t, miss = triggers(G[col], old)
                tr[col] = {"triggered": t, "not_checkable": miss}
                anytrig |= bool(t)
            G["triggers"] = tr
            S[f"{ck}/object"] = G
        res["slices"][sl] = S
    res["TRIGGER"] = anytrig
    json.dump(res, open(a.out, "w"), indent=1)
    for sl, S in res["slices"].items():
        for g, G in S.items():
            if g == "old_normal":
                continue
            for col in ("kp20", "kp20_G2", "kp20_GTfc2", "old_normal"):
                c = G[col]
                print(f"{sl:6s} {g:11s} {col:11s} " + " ".join(f"{k}={(round(v, 4) if isinstance(v, float) else v)}" for k, v in c.items()))
            print(f"{sl:6s} {g:11s} triggers {json.dumps(G['triggers'])}")
    print("TRIGGER", anytrig)


key_mode() if a.mode == "key" else summ_mode()
