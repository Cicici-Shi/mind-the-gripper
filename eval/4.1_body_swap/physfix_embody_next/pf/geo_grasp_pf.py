"""physfix armature rulings 2026-11-07 (astra_physfix_armature_rulings_2026-11-07.md Q2): geometric calibration of the
Robotiq-85 proprio / twin-opening map and the outcome-blind grasp-plausibility gate, XML armature config.
Physics only (no GL, no policy, no outcome of any pf_ cell). Code under test: --code (embody_next, unchanged).

Variants: panda (Panda, reference), a_kp100 (PandaR85 GRP_KP=100 R85_ARMATURE=1e-3 = XML main config), kp100 (old, no
armature), kp20 (stock Robotiq kp 20 = G'' physics).

Quantities (all per sim step, after BodyEnv.step):
  pad net gap = distance between the facing surfaces of the two pad collision boxes along the hand opening axis (hand
    y; the pads separate along it): (c_+ - c_-) . u - h_+(u) - h_-(u), h(u) = sum_i size_i |R_i . u| (box support).
    R85 pads: gripper0_{left,right}_fingerpad_collision (innermost distal surface; fingertip boxes sit 0.3 mm further
    out, probe 2026-11-07); fingertip gap also stored. Panda pads: gripper0_finger{1,2}_pad_collision.
  pad midpoint and each pad centre in the grip-site frame (twin pad alignment uses site z: R85 -10.0, Panda -3.6 mm).
  finger_joint, the 6 linkage joints, and the old map values f_kp20 / f_kp100 (BodyEnv._grp_proprio / PANDA_OPEN).
--mode sweep : free air from init 0, binary command open 20, close 120, open 60, close 60.
--mode grasp : scripted pick of the task's target object (object suite: obj_of_interest[0], one object per task;
    spatial: akita_black_bowl_1, grasped by its rim wall), steps: rise / traverse above the grasp point, descend, close
    30, lift 0.10 m, hold 30. Arm by OSC delta actions toward the target (pad midpoint), orientation held at the start
    orientation. Grasp point: object items = AABB centre (x, y) of its collision geoms, z = max(top - 25 mm, bottom +
    gripper-reach-below-pad-midpoint + 4 mm); if the object's extent along the opening axis > 68 mm it is turned 90 deg
    about z before the approach (logged). Bowl: radial midpoint of the rim wall (collision vertices within 10 mm of the
    top) along the opening axis, z = top - 15 mm (same floor rule).

===================== PRE-REGISTERED GATE (written 2026-11-07 before any grasp run; not to be edited after) =========
Per trial (suite, task, init, variant):
  T1 dual-pad contact: in the last 20 steps of the hold phase, BOTH pad collision geoms touch the target object in
     >= 90 % of the steps (pad geoms only; fingertip / other link contacts are reported, not counted).
  T2 carried: object root z at the end >= its z before closing + 0.07 m (lift commanded 0.10 m), AND slip = max over
     lift + hold of |r(t) - r(lift start)| <= 10 mm, r = object position in the hand frame relative to the pad midpoint.
  T3 penetration: max penetration (-dist) of gripper-object contacts from close start to the end <= 5 mm.
  T4 no breakdown: finger_joint within [-0.5, 1.2] every step (R85), BADQACC count unchanged, no arm-joint step
     |dq| > pi/2.
  trial PASS = T1 and T2 and T3 and T4.
Validity (script check): a (suite, task, init) is valid iff the Panda trial passes T1-T4 with the same script.
Object verdict for a variant: object script-invalid if it has < 2 valid inits; else PASS iff the variant passes
  >= ceil(2/3 * n_valid) of its valid inits, FAIL otherwise. Objects: the bowl (spatial tasks 0,2,3,6 x inits 0,1,2 =
  12 trials) and the 10 object-suite targets (tasks 0-9 x inits 0,1,2).
GATE (a_kp100 only; kp20 and kp100 printed as reference columns, never gate):
  FAIL          if any valid object FAILs;
  PASS          else if the bowl is valid and <= 2 of the 10 object-suite targets are script-invalid;
  INCONCLUSIVE  otherwise (script cannot test enough objects; stop and report, never treated as PASS).
=====================================================================================================================
Usage: python geo_grasp_pf.py --mode sweep|grasp --suite S --task T --init I --variant V --out FILE.json
       python geo_grasp_pf.py --mode summ --indir DIR --out summary.json
"""
import argparse, glob, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.path.expanduser("$PROJ_ROOT/embody_next"))
ap.add_argument("--mode", required=True, choices=["sweep", "grasp", "summ"])
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--init", type=int, default=0)
ap.add_argument("--variant", default="a_kp100")
ap.add_argument("--indir")
ap.add_argument("--out", required=True)
args = ap.parse_args()

VAR = {"panda": ("Panda", {}), "a_kp100": ("PandaR85", {"GRP_KP": "100", "R85_ARMATURE": "1e-3"}),
       "kp100": ("PandaR85", {"GRP_KP": "100"}), "kp20": ("PandaR85", {})}
PANDA_OPEN = 0.039
GJ = ["finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint", "right_outer_knuckle_joint",
      "right_inner_finger_joint", "right_inner_knuckle_joint"]


def box_h(m, d, g, u):
    R = d.geom_xmat[g].reshape(3, 3)
    return float(np.sum(m.geom_size[g] * np.abs(R.T @ u)))


def geom_verts(m, d, g):
    """World-frame points bounding geom g (mesh vertices; box corners; others by their bounding box)."""
    R, p, s, t = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
    if t == 7:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.asarray(m.mesh_vert[a:a + n], float)
    elif t == 6:
        v = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    elif t == 2:
        v = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[0], s[0]) for sz in (-s[0], s[0])])
    elif t in (3, 5):  # capsule / cylinder: radius s0, half length s1 (+ radius for capsule)
        hz = s[1] + (s[0] if t == 3 else 0)
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        v = np.array([[s[0] * np.cos(a_), s[0] * np.sin(a_), z] for a_ in ang for z in (-hz, hz)])
    elif t == 4:
        v = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
    else:
        return np.zeros((0, 3))
    return p + v @ R.T


if args.mode != "summ":
    for k in ("R85_ARMATURE", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "BODY"):
        os.environ.pop(k, None)
    os.environ.update(COPY_GRIP_CMD="1", ORIGINAL_ONLY="1")
    body, envv = VAR[args.variant]
    os.environ.update(envv)
    sys.path.insert(0, args.code)
    import mujoco  # noqa
    import embody  # noqa
    assert os.path.abspath(embody.__file__).startswith(os.path.abspath(args.code))
    from embody.body_env import BodyEnv  # noqa
    from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
    from libero.libero import get_libero_path  # noqa
    from libero.libero.envs.env_wrapper import ControlEnv  # noqa

    suite = OriginalObjectSuite() if args.suite == "libero_object" else OriginalSpatialSuite()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    np.random.seed(7)
    ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    be = BodyEnv(ref, bddl, body, 256, 10, log=lambda m_: None, render=False)
    be.reset()
    be.set_init_state(suite.get_task_init_states(args.task)[args.init])
    sim = be.env.sim
    m, d = sim.model, sim.data
    mr, dr = m._model, d._data
    rb = be.env.robots[0]
    r85 = body == "PandaR85"
    hb = m.body_name2id(rb.robot_model.eef_name)
    if r85:
        PADS = (m.geom_name2id("gripper0_right_fingerpad_collision"), m.geom_name2id("gripper0_left_fingerpad_collision"))
        TIPS = (m.geom_name2id("gripper0_right_fingertip_collision"), m.geom_name2id("gripper0_left_fingertip_collision"))
        gi = [m.get_joint_qpos_addr("gripper0_" + j) for j in GJ]
    else:
        PADS = (m.geom_name2id("gripper0_finger1_pad_collision"), m.geom_name2id("gripper0_finger2_pad_collision"))
        TIPS = None
        gi = list(rb._ref_gripper_joint_pos_indexes)
    GRIP = [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    arm_i = list(rb._ref_joint_pos_indexes)

    def badqacc():
        return int(dr.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    def geo():
        Rh = d.xmat[hb].reshape(3, 3)
        u = Rh[:, 1]
        cp, cm = d.geom_xpos[PADS[0]], d.geom_xpos[PADS[1]]
        sgn = 1.0 if (cp - cm) @ u > 0 else -1.0  # PADS[0] expected on +y
        gap = sgn * (cp - cm) @ u - box_h(m, d, PADS[0], u) - box_h(m, d, PADS[1], u)
        ps, Rs = d.site_xpos[rb.eef_site_id], d.site_xmat[rb.eef_site_id].reshape(3, 3)
        out = {"gap": float(gap), "sgn": sgn, "mid_site": (Rs.T @ ((cp + cm) / 2 - ps)).tolist(),
               "pad_p_site": (Rs.T @ (cp - ps)).tolist(), "pad_m_site": (Rs.T @ (cm - ps)).tolist(),
               "mid_w": ((cp + cm) / 2).tolist()}
        if TIPS:
            tp, tm = d.geom_xpos[TIPS[0]], d.geom_xpos[TIPS[1]]
            out["gap_tip"] = float(sgn * (tp - tm) @ u - box_h(m, d, TIPS[0], u) - box_h(m, d, TIPS[1], u))
        q = d.qpos[gi].copy()
        out["q"] = q.round(6).tolist()
        if r85:
            out["f_kp20"] = float(be._grp_proprio(q, "kp20")[0] / PANDA_OPEN)
            out["f_kp100"] = float(be._grp_proprio(q, "kp100")[0] / PANDA_OPEN)
        return out

    res = {"mode": args.mode, "suite": args.suite, "task": args.task, "init": args.init, "variant": args.variant,
           "armature": (mr.dof_armature[[m.jnt_dofadr[m.joint_name2id("gripper0_" + j)] for j in GJ]].tolist() if r85 else [])}
    q0 = badqacc()

if args.mode == "sweep":
    seq = [-1.0] * 20 + [1.0] * 120 + [-1.0] * 60 + [1.0] * 60
    rows = []
    for a in seq:
        be.step([0, 0, 0, 0, 0, 0, a])
        rows.append(geo())
    res["rows"] = rows
    res["seq"] = "open20 close120 open60 close60"
    res["badqacc"] = badqacc() - q0
    be.close()
    json.dump(res, open(args.out, "w"))
    print("done", args.out)

elif args.mode == "grasp":
    import robosuite.utils.transform_utils as T  # noqa
    e = be.env.env
    oname = "akita_black_bowl_1" if args.suite == "libero_spatial" else e.obj_of_interest[0]
    obj = e.objects_dict[oname]
    oroot = m.body_name2id(obj.root_body)

    def in_sub(b):
        while b > 0:
            if b == oroot:
                return True
            b = m.body_parentid[b]
        return False

    OG = [g for g in range(m.ngeom) if in_sub(m.geom_bodyid[g]) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ojn = obj.joints[0]
    oqa = m.get_joint_qpos_addr(ojn)[0]

    def overts():
        return np.concatenate([geom_verts(m, d, g) for g in OG])

    def hand_u():
        return d.xmat[hb].reshape(3, 3)[:, 1]

    # settle open
    for _ in range(10):
        be.step([0, 0, 0, 0, 0, 0, -1])
    V = overts()
    uh = hand_u().copy(); uh[2] = 0; uh /= np.linalg.norm(uh)
    vh = np.array([-uh[1], uh[0], 0.0])
    ctr = (V.min(0) + V.max(0)) / 2
    ext_u = float(np.ptp(V @ uh)); ext_v = float(np.ptp(V @ vh))
    rotated = False
    bowl = oname.startswith("akita_black_bowl")
    if not bowl and ext_u > 0.068 and ext_v < ext_u:
        qz = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])  # wxyz, 90 deg about z
        pos, quat = d.qpos[oqa:oqa + 3].copy(), d.qpos[oqa + 3:oqa + 7].copy()
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        d.qpos[oqa:oqa + 3] = ctr + Rz @ (pos - ctr)
        qn = np.zeros(4); mujoco.mju_mulQuat(qn, qz, quat)
        d.qpos[oqa + 3:oqa + 7] = qn
        d.qvel[m.jnt_dofadr[m.joint_name2id(ojn)]:m.jnt_dofadr[m.joint_name2id(ojn)] + 6] = 0
        sim.forward()
        for _ in range(20):
            be.step([0, 0, 0, 0, 0, 0, -1])
        rotated = True
        V = overts(); ctr = (V.min(0) + V.max(0)) / 2
        ext_u, ext_v = float(np.ptp(V @ uh)), float(np.ptp(V @ vh))
    top, bot = float(V[:, 2].max()), float(V[:, 2].min())
    # how far the gripper reaches below the pad midpoint (open state)
    pm = np.array(geo()["mid_w"])
    lowz = min(float(geom_verts(m, d, g)[:, 2].min()) for g in GRIP if m.geom_type[g] in (6, 7))
    reach = float(pm[2] - lowz)
    if bowl:
        rim = V[V[:, 2] > top - 0.010]
        rad, perp = (rim - ctr) @ uh, (rim - ctr) @ vh
        rpos = rad[(rad > 0) & (np.abs(perp) < 0.01)]  # rim vertices on the +u ray
        r_out, r_in = float(rpos.max()), float(rpos.min())
        gxy = ctr[:2] + ((r_out + r_in) / 2) * uh[:2]
        gz = max(top - 0.015, bot + reach + 0.004)
        res["bowl_rim"] = {"r_out": r_out, "r_in": r_in, "wall_mm": (r_out - r_in) * 1000}
    else:
        gxy = ctr[:2]
        gz = max(top - 0.025, bot + reach + 0.004)
    G = np.array([gxy[0], gxy[1], gz])
    zab = top + 0.08 + 0.0
    R0 = d.xmat[hb].reshape(3, 3).copy()
    res.update(obj=oname, rotated=rotated, ext_u_mm=ext_u * 1000, ext_v_mm=ext_v * 1000, top=top, bot=bot,
               reach_mm=reach * 1000, grasp_pt=G.tolist())
    rows, phase_of = [], []
    qprev = d.qpos[arm_i].copy()
    max_dq = 0.0

    def act(tgt, grip, gain=1.0):
        p = np.array(geo()["mid_w"])
        dx = np.clip((tgt - p) / 0.05 * gain, -1, 1)
        R = d.xmat[hb].reshape(3, 3)
        rv = T.quat2axisangle(T.mat2quat(R0 @ R.T))
        dr_ = np.clip(rv / 0.5, -1, 1)
        return list(dx) + list(dr_) + [grip], float(np.linalg.norm(tgt - p))

    def contacts():
        out = {"pad": [0, 0], "tip": [0, 0], "other": 0, "pen": 0.0, "fn": 0.0}
        f6 = np.zeros(6)
        for i in range(dr.ncon):
            c = dr.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in OG and g2 in GRIP:
                gg_ = g2
            elif g2 in OG and g1 in GRIP:
                gg_ = g1
            else:
                continue
            out["pen"] = max(out["pen"], -float(c.dist))
            mujoco.mj_contactForce(mr, dr, i, f6)
            out["fn"] += float(f6[0])
            if gg_ == PADS[0]:
                out["pad"][0] += 1
            elif gg_ == PADS[1]:
                out["pad"][1] += 1
            elif TIPS and gg_ == TIPS[0]:
                out["tip"][0] += 1
            elif TIPS and gg_ == TIPS[1]:
                out["tip"][1] += 1
            else:
                out["other"] += 1
        return out

    def run_phase(name, tgt, grip, nmax, tol=None, gain=1.0):
        global qprev, max_dq
        for k in range(nmax):
            a, err = act(tgt, grip, gain)
            if tol is not None and err < tol:
                break
            be.step(a)
            qa = d.qpos[arm_i].copy()
            max_dq = max(max_dq, float(np.abs(qa - qprev).max()))
            qprev = qa
            r = geo()
            Rh = d.xmat[hb].reshape(3, 3)
            r["obj_rel"] = (Rh.T @ (d.xpos[oroot] - np.array(r["mid_w"]))).tolist()
            r["obj_z"] = float(d.xpos[oroot][2])
            r["con"] = contacts()
            r["err"] = err
            rows.append(r)
            phase_of.append(name)

    p0 = np.array(geo()["mid_w"])
    run_phase("rise", np.array([p0[0], p0[1], max(p0[2], zab)]), -1, 60, tol=0.004)
    run_phase("traverse", np.array([G[0], G[1], zab]), -1, 100, tol=0.003)
    run_phase("descend", G, -1, 100, tol=0.002, gain=0.6)
    run_phase("settle", G, -1, 5)
    z_before = float(d.xpos[oroot][2])
    i_close = len(rows)
    run_phase("close", G, 1, 30)
    i_lift = len(rows)
    run_phase("lift", G + np.array([0, 0, 0.10]), 1, 60, tol=0.003)
    run_phase("hold", G + np.array([0, 0, 0.10]), 1, 30)
    res["badqacc"] = badqacc() - q0
    res["phases"] = phase_of
    res["rows"] = rows
    hold = [r for r, ph in zip(rows, phase_of) if ph == "hold"][-20:]
    t1 = float(np.mean([r["con"]["pad"][0] > 0 and r["con"]["pad"][1] > 0 for r in hold])) if hold else 0.0
    zgain = float(rows[-1]["obj_z"] - z_before)
    r_l = np.array(rows[i_lift]["obj_rel"]) if i_lift < len(rows) else None
    slip = float(max(np.linalg.norm(np.array(r["obj_rel"]) - r_l) for r in rows[i_lift:])) if r_l is not None else 1e9
    pen = float(max(r["con"]["pen"] for r in rows[i_close:]))
    fj = np.array([r["q"][0] for r in rows]) if r85 else np.zeros(1)
    T1, T2, T3 = t1 >= 0.9, zgain >= 0.07 and slip <= 0.010, pen <= 0.005
    T4 = bool(fj.min() >= -0.5 and fj.max() <= 1.2 and res["badqacc"] == 0 and max_dq <= math.pi / 2)
    res["metrics"] = {"T1_dualpad_frac": t1, "zgain_m": zgain, "slip_mm": slip * 1000, "pen_mm": pen * 1000,
                      "fj_min": float(fj.min()), "fj_max": float(fj.max()), "max_arm_dq": max_dq,
                      "hold_tip_frac": float(np.mean([r["con"]["tip"][0] > 0 and r["con"]["tip"][1] > 0 for r in hold])) if hold else 0.0,
                      "hold_any_side": [float(np.mean([r["con"]["pad"][s] + r["con"]["tip"][s] > 0 for r in hold])) if hold else 0.0 for s in (0, 1)],
                      "hold_gap_mm": float(np.median([r["gap"] for r in hold]) * 1000) if hold else None,
                      "hold_fn": float(np.median([r["con"]["fn"] for r in hold])) if hold else None,
                      "close_end_gap_mm": float(rows[i_lift - 1]["gap"] * 1000) if i_lift > 0 else None,
                      "descend_err_mm": float(rows[i_close - 1]["err"] * 1000) if i_close > 0 else None}
    res["T"] = {"T1": bool(T1), "T2": bool(T2), "T3": bool(T3), "T4": T4}
    res["PASS"] = bool(T1 and T2 and T3 and T4)
    be.close()
    json.dump(res, open(args.out, "w"))
    print("done", args.out, res["PASS"], res["T"], {k: round(v, 3) if isinstance(v, float) else v for k, v in res["metrics"].items()})

else:  # summ
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.indir, "*.json"))) if not f.endswith("summary.json")]
    S = {"sweep": {}, "grasp": {}}
    # Panda: pad gap vs finger qpos sum
    pq, pg = [], []
    for r in R:
        if r["mode"] == "sweep" and r["variant"] == "panda":
            for x in r["rows"]:
                pq.append(x["q"][0] - x["q"][1]); pg.append(x["gap"])
    a_, b_ = np.polyfit(pq, pg, 1)
    resid = np.array(pg) - (a_ * np.array(pq) + b_)
    S["panda_fit"] = {"gap = a*(q1-q2)+b": [float(a_), float(b_)], "max_resid_mm": float(np.abs(resid).max() * 1000),
                      "gap_open_mm": float(max(pg) * 1000), "gap_closed_mm": float(min(pg) * 1000),
                      "mid_site_z_mm": [float(min(x["mid_site"][2] for r in R if r["mode"] == "sweep" and r["variant"] == "panda" for x in r["rows"]) * 1000),
                                        float(max(x["mid_site"][2] for r in R if r["mode"] == "sweep" and r["variant"] == "panda" for x in r["rows"]) * 1000)]}

    def f_geo(gap):
        return float(np.clip(((gap - b_) / a_) / (2 * PANDA_OPEN), 0, 1))

    for r in R:
        if r["mode"] != "sweep" or r["variant"] == "panda":
            continue
        rows = r["rows"]
        gap = np.array([x["gap"] for x in rows]); fj = np.array([x["q"][0] for x in rows])
        fg = np.array([f_geo(g) for g in gap]); f100 = np.array([x["f_kp100"] for x in rows]); f20 = np.array([x["f_kp20"] for x in rows])
        mz = np.array([x["mid_site"][2] for x in rows])
        key = f"{r['suite']}:{r['task']}:{r['variant']}"
        S["sweep"][key] = {
            "armature": r["armature"], "badqacc": r["badqacc"],
            "fj_open": float(fj[19]), "fj_closed_40": float(fj[59]), "fj_closed_120": float(fj[139]), "fj_last20_ptp": float(np.ptp(fj[120:140])),
            "gap_open_mm": float(gap[19] * 1000), "gap_closed_40_mm": float(gap[59] * 1000), "gap_closed_120_mm": float(gap[139] * 1000),
            "gap_min_mm": float(gap.min() * 1000), "gap_reopen_mm": float(gap[199] * 1000), "gap_closed2_mm": float(gap[259] * 1000),
            "tipgap_closed_mm": float(rows[139]["gap_tip"] * 1000),
            "f_geo_closed": float(fg[139]), "f_kp100_closed": float(f100[139]), "f_kp20_closed": float(f20[139]),
            "err_kp100_vs_geo": {"max": float(np.abs(f100 - fg).max()), "p90": float(np.percentile(np.abs(f100 - fg), 90))},
            "err_kp20_vs_geo": {"max": float(np.abs(f20 - fg).max()), "p90": float(np.percentile(np.abs(f20 - fg), 90))},
            "mid_site_z_mm": [float(mz.min() * 1000), float(mz.max() * 1000)],
            "mid_site_lat_mm_max": float(max(math.hypot(x["mid_site"][0], x["mid_site"][1]) for x in rows) * 1000),
            "curve_close1": [[round(float(fj[i]), 4), round(float(gap[i] * 1000), 2)] for i in range(19, 60, 2)],
            "curve_open1": [[round(float(fj[i]), 4), round(float(gap[i] * 1000), 2)] for i in range(139, 180, 2)]}
    # grasp
    T = {}
    for r in R:
        if r["mode"] == "grasp":
            T[(r["suite"], r["task"], r["init"], r["variant"])] = r
    objs = {}
    for (s, t, i, v), r in T.items():
        objs.setdefault(r["obj"], set()).add((s, t, i))
    verdict = {}
    for o, trials in sorted(objs.items()):
        valid = [k for k in trials if (k + ("panda",)) in T and T[k + ("panda",)]["PASS"]]
        verdict[o] = {"n_trials": len(trials), "n_valid": len(valid)}
        for v in ("a_kp100", "kp20", "kp100"):
            npass = sum(1 for k in valid if (k + (v,)) in T and T[k + (v,)]["PASS"])
            if len(valid) < 2:
                vd = "script-invalid"
            else:
                vd = "PASS" if npass >= math.ceil(2 / 3 * len(valid)) else "FAIL"
            fails = {f"{k[1]}:{k[2]}": T[k + (v,)]["T"] for k in valid if (k + (v,)) in T and not T[k + (v,)]["PASS"]}
            hold_gap = [T[k + (v,)]["metrics"]["hold_gap_mm"] for k in trials if (k + (v,)) in T]
            ce_gap = [T[k + (v,)]["metrics"]["close_end_gap_mm"] for k in trials if (k + (v,)) in T]
            verdict[o][v] = {"verdict": vd, "n_pass_valid": npass, "fails": fails,
                             "close_end_gap_mm": [round(x, 1) if x is not None else None for x in ce_gap],
                             "hold_gap_mm": [round(x, 1) if x is not None else None for x in hold_gap]}
        verdict[o]["ext_u_mm"] = sorted({round(T[k + ("panda",)]["ext_u_mm"], 1) for k in trials if (k + ("panda",)) in T})
        verdict[o]["panda_hold_gap_mm"] = [round(T[k + ("panda",)]["metrics"]["hold_gap_mm"] or -1, 1) for k in trials if (k + ("panda",)) in T]
    S["grasp"] = verdict
    bowl = [o for o in verdict if o.startswith("akita")]
    others = [o for o in verdict if not o.startswith("akita")]
    gate = {}
    for v in ("a_kp100", "kp20", "kp100"):
        vds = {o: verdict[o][v]["verdict"] for o in verdict}
        n_inv = sum(1 for o in others if vds[o] == "script-invalid")
        if any(x == "FAIL" for x in vds.values()):
            g = "FAIL"
        elif bowl and vds[bowl[0]] != "script-invalid" and n_inv <= 2 and len(others) == 10:
            g = "PASS"
        else:
            g = "INCONCLUSIVE"
        gate[v] = {"gate": g, "objects": vds, "n_invalid_object_suite": n_inv}
    S["gate"] = gate
    # grasp-state map check: proprio maps vs pad gap in blocked states (a_kp100, close-end and hold)
    errs = {"kp100": [], "kp20": []}
    for (s, t, i, v), r in T.items():
        if v != "a_kp100":
            continue
        for x, ph in zip(r["rows"], r["phases"]):
            if ph in ("hold", "lift"):
                fg = f_geo(x["gap"])
                errs["kp100"].append(abs(x["f_kp100"] - fg)); errs["kp20"].append(abs(x["f_kp20"] - fg))
    S["blocked_map_err_a_kp100"] = {k: {"max": float(max(v)), "p90": float(np.percentile(v, 90)), "n": len(v)} for k, v in errs.items() if v}
    # pad-midpoint z in the site frame in blocked states (twin alignment table check)
    for v in ("a_kp100", "kp20", "panda"):
        z = [x["mid_site"][2] for (s, t, i, vv), r in T.items() if vv == v for x, ph in zip(r["rows"], r["phases"]) if ph in ("close", "lift", "hold")]
        lat = [math.hypot(x["mid_site"][0], x["mid_site"][1]) for (s, t, i, vv), r in T.items() if vv == v for x, ph in zip(r["rows"], r["phases"]) if ph in ("close", "lift", "hold")]
        if z:
            S.setdefault("blocked_mid_site", {})[v] = {"z_mm": [float(min(z) * 1000), float(np.median(z) * 1000), float(max(z) * 1000)],
                                                       "lat_mm_p90_max": [float(np.percentile(lat, 90) * 1000), float(max(lat) * 1000)]}
    json.dump(S, open(args.out, "w"), indent=1)
    print(json.dumps({"panda_fit": S["panda_fit"], "gate": S["gate"]}, indent=1))
    for k, v in S["sweep"].items():
        print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("curve")})
    for o, v in verdict.items():
        print(o, json.dumps(v))
    print("blocked_map_err", S.get("blocked_map_err_a_kp100"))
    print("blocked_mid_site", S.get("blocked_mid_site"))
