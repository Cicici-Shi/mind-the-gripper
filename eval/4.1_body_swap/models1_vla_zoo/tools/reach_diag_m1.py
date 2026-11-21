"""models1 twin reachability check (Fable ruling 2026-11-05, section 3 item 4 last point), CPU only, no rendering.

For OFTi dumps (npz keep the physical qpos of every policy step) rebuild the twin IK target of each painted frame:
physical PandaR85 grip-site pose from the recorded qpos, shifted along site z by BodyEnv._pad_shift("r85","panda")
= -6.4 mm, exactly as embody_toy BodyEnv._twin_paint_inner does (TWIN_CAM_SYNC=1 -> target = FK of qpos).
  verify : replay BodyEnv's own IK chain (ik_site_ns warm start, gain 0, >1e-3 m retry from rest / init_qpos /
           _seed_postures) over whole episodes and compare with the recorded twin_pos_err_mm / twin_rot_err_deg.
  diag   : for frames whose recorded residual > 1 mm (plus control frames < 0.05 mm), as embody/twin_reach_diag.py:
           full-pose IK from 50 uniform random in-range seeds (ik_site, 600 iters) and position-only IK; report
           best residuals, Panda joints within 1 deg of a limit, physical-arm joints near limits, grip yaw / tilt.
Never reads the `success` field of any record (npz members are loaded lazily by name).
Usage: python reach_diag_m1.py verify|diag --model oft_obj --suite libero_object --cells obj_g_ta_fc,obj_g_tfc --offsets 0
"""
import argparse, glob, json, os, re, sys, time
from multiprocessing import Pool

import numpy as np

EMB = os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_toy"))
sys.path.insert(0, EMB)
Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
OUT = os.path.join(Z, "out", "reach_diag_m1")
DZ = -0.0064  # BodyEnv._pad_shift("r85", "panda"), R85_PAD_BACK


def keys_of(model, cell, offsets, trials=5):
    keys = {(t, e) for t in range(10) for o in offsets for e in range(o, o + trials)}
    out = {}
    for f in glob.glob(os.path.join(Z, "runs", model, cell, "t*_e*_i*.npz")):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        if k in keys:
            out[k] = f
    return out


def load(f):
    z = np.load(f, allow_pickle=True)  # lazy: only the members named below are read
    r = {n: z[n] for n in ("task", "ep", "init", "qpos", "joint_names", "eef_pos", "eef_quat")}
    for n in ("twin_pos_err_mm", "twin_rot_err_deg"):  # non-twin cells (S, G): no residual, zeros
        r[n] = z[n] if n in z.files else np.zeros(len(r["qpos"]) + 1, np.float32)
    return r


_ENV = {}


def envs(suite, tid):
    if (suite, tid) in _ENV:
        return _ENV[(suite, tid)]
    from embody import robots  # noqa: F401  registers MountedPandaR85
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    task = benchmark.get_benchmark_dict()[suite]().get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    mk = lambda body: ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False,
                                 has_offscreen_renderer=False, use_camera_obs=False)
    ph, tw = mk("PandaR85"), mk("Panda")
    ph.reset(); tw.reset()
    _ENV[(suite, tid)] = (ph, tw)
    return ph, tw


def target(ph, qpos_row):
    from embody import align
    import mujoco
    m, d = align._raw(ph.sim)
    assert m.nq == len(qpos_row), (m.nq, len(qpos_row))
    d.qpos[:] = qpos_row
    mujoco.mj_kinematics(m, d)
    p, R = align.site_pose(ph.sim, ph.robots[0].eef_site_id)
    return p + R @ np.array([0.0, 0.0, DZ]), R, p


def twin_resid(tw, q, p_t, R_t):
    from embody import align
    import mujoco
    m, d = align._raw(tw.sim)
    saved = d.qpos.copy()
    d.qpos[tw.robots[0]._ref_joint_pos_indexes] = q
    mujoco.mj_kinematics(m, d)
    p, R = align.site_pose(tw.sim, tw.robots[0].eef_site_id)
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(p - p_t)) * 1000, float(align.rot_err_deg(R, R_t))


class _SeedHost:  # just enough of BodyEnv for its own _seed_postures (same code, same library)
    twin_body, twin_seed_phys = "Panda", False

    def log(self, s):
        print(s, flush=True)


def seed_postures(p):
    from embody.body_env import BodyEnv
    return BodyEnv._seed_postures(_SeedHost(), p)


def ref_rest(suite, k, tw):
    """Reference Panda joints after the wait steps = qpos[0] of the same key's native run (twin _twin_start q0)."""
    d = "obj_src" if suite == "libero_object" else "src"
    fs = glob.glob(os.path.join(Z, "runs", "oft_obj" if suite == "libero_object" else "oft_sp", d, f"t{k[0]}_e{k[1]}_i*.npz"))
    z = np.load(fs[0], allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    m = tw.sim.model
    idx = [m.get_joint_qpos_addr(f"robot0_joint{j}") for j in range(1, 8)]
    assert names[:7] == [f"robot0_joint{j}" for j in range(1, 8)]
    return np.asarray(z["qpos"][0], float)[idx], fs[0]


def verify_key(job):
    suite, cell, k, f = job
    from embody import align
    r = load(f)
    ph, tw = envs(suite, int(r["task"]))
    rest, srcf = ref_rest(suite, k, tw)
    init = np.asarray(tw.robots[0].init_qpos, float)
    q = rest.copy()
    n = len(r["qpos"])
    pe_rec, re_rec = np.asarray(r["twin_pos_err_mm"], float), np.asarray(r["twin_rot_err_deg"], float)
    pe, re_, eefd = [], [], []
    for i in range(n):
        p_t, R_t, p_site = target(ph, r["qpos"][i])
        eefd.append(float(np.linalg.norm(p_site - r["eef_pos"][i])) * 1000)
        q, ep, er = align.ik_site_ns(tw, p_t, R_t, q, rest, gain=0.0)
        if ep > 1e-3:
            for seed in [rest, init] + seed_postures(p_t):
                q2, ep2, er2 = align.ik_site_ns(tw, p_t, R_t, seed, rest, gain=0.0)
                if ep2 < ep:
                    q, ep, er = q2, ep2, er2
                if ep < 1e-5:
                    break
        a, b = twin_resid(tw, q, p_t, R_t)
        pe.append(a); re_.append(b)
    pe, re_ = np.array(pe), np.array(re_)
    dp = np.abs(pe - pe_rec[:n]); dr = np.abs(re_ - re_rec[:n])
    over_rec = set(np.nonzero(pe_rec[:n] > 1)[0].tolist()); over_new = set(np.nonzero(pe > 1)[0].tolist())
    return {"cell": cell, "key": list(k), "src_ref": os.path.basename(srcf),
            "pos_absdiff_max_mm": float(dp.max()), "pos_absdiff_p99_mm": float(np.percentile(dp, 99)),
            "rot_absdiff_max_deg": float(dr.max()), "bitwise_float32_equal_pos": int(np.sum(pe.astype(np.float32) == pe_rec[:n].astype(np.float32))),
            "over1_rec": len(over_rec), "over1_replay": len(over_new), "over1_same": len(over_rec & over_new),
            "rec_max_mm": float(pe_rec[:n].max()), "replay_max_mm": float(pe.max()),
            "eef_obs_vs_fk_mm_max": float(max(eefd)), "eef_obs_vs_fk_mm_med": float(np.median(eefd))}


def diag_frame(job):
    suite, cell, k, f, i, kind, seed = job
    from embody import align
    import robosuite.utils.transform_utils as T  # noqa: F401
    r = load(f)
    ph, tw = envs(suite, int(r["task"]))
    p_t, R_t, _ = target(ph, r["qpos"][i])
    m = tw.sim.model
    jid = [m.joint_name2id(j) for j in tw.robots[0].robot_joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    rng = np.random.default_rng(seed)
    best = None
    n_full_ok = 0
    for _ in range(50):
        q = align.ik_site(tw, p_t, R_t, rng.uniform(lo, hi), iters=600)
        a, b = twin_resid(tw, q, p_t, R_t)
        n_full_ok += a <= 1.0
        if best is None or a + b < best[0]:
            best = (a + b, q, a, b)
    # position-only IK (twin_reach_diag.pos_only_ik), from the best full-pose solution, then from 10 random seeds
    import mujoco
    mm, d = align._raw(tw.sim)
    qi = np.array(tw.robots[0]._ref_joint_pos_indexes); di = np.array(tw.robots[0]._ref_joint_vel_indexes)
    sid = tw.robots[0].eef_site_id

    def pos_only(q0, iters=600, damping=1e-3):
        saved = d.qpos.copy(); q = np.array(q0, float); jacp = np.zeros((3, mm.nv))
        for _ in range(iters):
            d.qpos[qi] = q
            mujoco.mj_kinematics(mm, d); mujoco.mj_comPos(mm, d)
            p, _ = align.site_pose(tw.sim, sid)
            e = p_t - p
            if np.linalg.norm(e) < 1e-5:
                break
            mujoco.mj_jacSite(mm, d, jacp, None, sid)
            J = jacp[:, di]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), e), lo, hi)
        d.qpos[qi] = q; mujoco.mj_kinematics(mm, d)
        p, _ = align.site_pose(tw.sim, sid)
        d.qpos[:] = saved; mujoco.mj_forward(mm, d)
        return float(np.linalg.norm(p_t - p)) * 1000
    e_pos = pos_only(best[1])
    if e_pos > 0.01:
        e_pos = min([e_pos] + [pos_only(rng.uniform(lo, hi)) for _ in range(10)])
    margin = np.degrees(np.minimum(best[1] - lo, hi - best[1]))
    at_lim = [j + 1 for j in range(7) if margin[j] < 1.0]
    # physical arm (same 7-joint Panda arm, same limits) at this frame
    pm = ph.sim.model
    pidx = [pm.get_joint_qpos_addr(f"robot0_joint{j}") for j in range(1, 8)]
    qph = np.asarray(r["qpos"][i], float)[pidx]
    pmar = np.degrees(np.minimum(qph - lo, hi - qph))
    yaw = float(np.degrees(np.arctan2(R_t[1, 0], R_t[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(-R_t[2, 2], -1, 1))))
    base = tw.sim.data.get_body_xpos(tw.robots[0].robot_model.root_body)
    return {"cell": cell, "key": list(k), "step": int(i), "query": i % 8 == 0, "kind": kind,
            "rec_pos_mm": float(r["twin_pos_err_mm"][i]), "rec_rot_deg": float(r["twin_rot_err_deg"][i]),
            "full_best_pos_mm": best[2], "full_best_rot_deg": best[3], "full_seeds_le1mm": int(n_full_ok),
            "posonly_mm": e_pos, "twin_joints_at_limit": at_lim, "twin_margin_deg": np.round(margin, 2).tolist(),
            "phys_margin_deg": np.round(pmar, 2).tolist(), "phys_joints_lt1deg": [j + 1 for j in range(7) if pmar[j] < 1.0],
            "phys_joints_lt5deg": [j + 1 for j in range(7) if pmar[j] < 5.0],
            "target_yaw_deg": yaw, "target_tilt_deg": tilt, "target_xyz": np.round(p_t, 4).tolist(),
            "target_dist_from_base_m": float(np.linalg.norm(p_t - base))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "diag", "scan"])
    ap.add_argument("--model", default="oft_obj")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--cells", default="obj_g_ta_fc,obj_g_tfc")
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--verify-keys", default="3", help="per cell: keys with most >1 mm frames to replay")
    ap.add_argument("--controls", type=int, default=15, help="per cell: frames < 0.05 mm as reachable controls")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    offs = [int(x) for x in a.offsets.split(",")]
    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or os.environ.get("SLURM_JOB_ID", "local")
    jobs, meta = [], {}
    for cell in a.cells.split(","):
        ks = keys_of(a.model, cell, offs)
        tot_all = tot_q = over_all = over_q = 0
        per_key = {}
        for k, f in sorted(ks.items()):
            r = load(f)
            pe = np.asarray(r["twin_pos_err_mm"], float)
            n = len(r["qpos"])
            bad = [i for i in np.nonzero(pe > 1)[0].tolist() if i < n]
            per_key[k] = (f, bad, n)
            tot_all += len(pe); over_all += int(np.sum(pe > 1))
            tot_q += len(pe[::8]); over_q += int(np.sum(pe[::8] > 1))
        meta[cell] = {"keys": len(ks), "shards": offs, "all_frames": tot_all, "all_over1": over_all,
                      "query_frames": tot_q, "query_over1": over_q,
                      "keys_with_over1": sum(1 for v in per_key.values() if v[1])}
        print(f"[{cell}] keys {len(ks)} shards {offs}: all-frame >1mm {over_all}/{tot_all} = {over_all/max(tot_all,1):.4%}; "
              f"query-frame (::8) >1mm {over_q}/{tot_q} = {over_q/max(tot_q,1):.4%}; keys with any >1mm {meta[cell]['keys_with_over1']}", flush=True)
        if a.mode == "scan":
            # numpy only: keys with a physics jump (any arm joint > 90 deg in one control step) and where the
            # >1 mm frames sit relative to it. Episode length is never printed.
            jumps, bad_after, bad_total, bad_in_jumpkeys, gap = 0, 0, 0, 0, []
            for k, (f, bad, n) in sorted(per_key.items()):
                q = np.asarray(load(f)["qpos"], float)[:, :7]
                dq = np.degrees(np.abs(np.diff(q, axis=0))).max(1)
                js = np.nonzero(dq > 90)[0]
                bad_total += len(bad)
                if len(js):
                    jumps += 1
                    bad_in_jumpkeys += len(bad)
                    bad_after += sum(1 for i in bad if i > js[0])
                    if bad:
                        gap.append(int(min(bad) - js[0]))
            meta[cell].update({"keys_with_arm_jump_gt90deg": jumps, "over1_frames_all": bad_total,
                               "over1_frames_in_jump_keys": bad_in_jumpkeys, "over1_frames_after_first_jump": bad_after,
                               "steps_from_jump_to_first_over1": gap})
            print(f"  scan {cell}: keys with an arm-joint jump > 90 deg/step {jumps}/{len(per_key)}; >1 mm frames {bad_total}, "
                  f"in those keys {bad_in_jumpkeys}, after the first jump {bad_after}; steps jump -> first >1 mm {gap}", flush=True)
            continue
        if a.mode == "verify":
            top = sorted(per_key.items(), key=lambda kv: -len(kv[1][1]))[: int(a.verify_keys)]
            jobs += [(a.suite, cell, k, v[0]) for k, v in top]
        else:
            rng = np.random.default_rng(1)
            seed = 0
            for k, (f, bad, n) in sorted(per_key.items()):
                for i in bad:
                    jobs.append((a.suite, cell, k, f, i, "over1", seed)); seed += 1
            pool = []
            for k, (f, bad, n) in sorted(per_key.items()):
                pe = np.asarray(load(f)["twin_pos_err_mm"], float)
                pool += [(k, f, i) for i in range(n) if pe[i] < 0.05]
            for j in rng.choice(len(pool), size=min(a.controls, len(pool)), replace=False):
                k, f, i = pool[j]
                jobs.append((a.suite, cell, k, f, int(i), "control", seed)); seed += 1
    if a.mode == "scan":
        out = os.path.join(OUT, f"scan_{a.model}_{tag}.json")
        json.dump({"meta": meta}, open(out, "w"), indent=1)
        print(f"wrote {out}")
        return
    # group by task so each worker builds its envs once
    jobs.sort(key=lambda j: (j[2][0], j[1], j[2][1]))
    t0 = time.time()
    fn = verify_key if a.mode == "verify" else diag_frame
    with Pool(a.nproc) as p:
        res = p.map(fn, jobs, chunksize=1)
    out = os.path.join(OUT, f"{a.mode}_{a.model}_{tag}.json")
    json.dump({"meta": meta, "rows": res, "embody_dir": EMB, "wall_s": time.time() - t0}, open(out, "w"), indent=1)
    for row in res:
        print(json.dumps(row), flush=True)
    print(f"wrote {out}  ({len(res)} rows, {time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
