"""Shared pieces of the toy1 cached-frame probes (protocol_toy1.md 5.3 / 5.4), 2026-11-03.

- OFT: frozen openvla-7b-oft (same loading and preprocessing as oft/run_libero_eval_toy.py); query() returns the raw
  chunk (un-normalised) and the processed chunk (process_action: binarised, sign-flipped gripper; positions unchanged).
- Scene: reference fixtures from the dump, same init state, BodyEnv.set_init_state, then the physical state of a
  frame = the dumped qpos (velocities 0, mj_forward; images depend on positions only).
- Frames (5.3): phase labels as analysis/probe_phase.py (approach / pre-grasp by grip site within 7.5 cm of the target
  bowl origin, before the first close command); only chunk boundaries (step % 8 == 0); the last 2 approach boundaries
  before the first pre-grasp step + 3 pre-grasp boundaries equally spaced (np.linspace over the pre-grasp boundaries,
  rounded); episodes with < 3 pre-grasp boundaries are skipped and the next episode of the task is taken.
"""
import glob, importlib.util, os, sys
import numpy as np

TOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOY)
sys.path.insert(0, os.path.join(TOY, "analysis"))
from stages import qpos_addr, task_objects  # noqa: E402
from probe_phase import phase_labels  # noqa: E402

_R = None


def oft():
    """(module, cfg, model, action_head, proprio_projector, noisy_action_projector, processor, resize_size)."""
    global _R
    if _R is None:
        spec = importlib.util.spec_from_file_location("rle_toy", os.path.join(TOY, "oft", "run_libero_eval_toy.py"))
        R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R)
        cfg = R.GenerateConfig(pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                               task_suite_name="libero_spatial")
        R.set_seed_everywhere(cfg.seed)
        parts = R.initialize_model(cfg)
        _R = (R, cfg, *parts, R.get_image_resize_size(cfg))
    return _R


def query(obs, lang):
    if os.environ.get("PROBE_DRY") == "1":  # render-path test without the policy (no GPU memory for OFT needed)
        return np.zeros((8, 7)), np.zeros((8, 7)), np.zeros(8)
    R, cfg, model, action_head, proprio_projector, noisy, processor, rs = oft()
    o, _ = R.prepare_observation(obs, rs)
    a = R.get_action(cfg, model, o, lang, processor=processor, action_head=action_head,
                     proprio_projector=proprio_projector, noisy_action_projector=noisy, use_film=cfg.use_film)
    raw = np.asarray(a, dtype=np.float64)
    proc = np.stack([R.process_action(np.array(x, copy=True), cfg.model_family) for x in raw])
    return raw, proc, np.asarray(o["state"], dtype=np.float64)


def set_ref_fixtures(rm, z):
    for i, n in enumerate(str(x) for x in z["ref_body_names"]):
        if not n:
            continue
        try:
            bid = rm.body_name2id(n)
        except (ValueError, KeyError):
            continue
        rm.body_pos[bid] = z["ref_body_pos"][i]
        rm.body_quat[bid] = z["ref_body_quat"][i]


def start_episode(env, z, inits):
    env.reset()
    set_ref_fixtures(env.ref.sim.model, z)
    return env.set_init_state(inits[int(z["init"])])


def physical_obs(env, qpos):
    """Physical state = qpos, then the plain physical observation (no twin paint, BodyEnv._fix applied)."""
    e = env.env  # OffScreenRenderEnv wrapper; the robosuite env is e.env (as BodyEnv._grp_init uses env.env)
    assert e.sim.data.qpos.shape == qpos.shape, (e.sim.data.qpos.shape, qpos.shape)
    e.sim.data.qpos[:] = qpos
    e.sim.data.qvel[:] = 0.0
    e.sim.forward()
    e.env._update_observables(force=True)
    return e.env._get_observations()


def select_frames(z, target, n_app=2, n_pre=3, rule="protocol"):
    """rule="protocol": protocol_toy1 5.3 as written. rule="preclose": the last n_app + n_pre chunk boundaries strictly
    before the first close command (any distance), each labelled with its probe_phase label (pending a decision:
    the protocol's pre-grasp window is < 1 chunk long in 200/200 SRC and 197/200 G episodes, WORKLOG_toy1 4.6)."""
    lab = phase_labels(z, target)
    if rule == "astra":
        # protocol_toy1_astra_probe_rule.md 2-3: t_c = first step whose recorded gripper command > 0; frames
        # t_c-5 .. t_c-1 (valid, distinct, any step), no padding; no close command -> missing (no replacement).
        # stratum by the un-intervened grip-site distance to the target bowl centre (bowl free-joint position, the
        # probe_phase / stages.py convention): < 7.5 cm main, 7.5-15, > 15 cm.
        close = np.nonzero(np.asarray(z["action"])[:, -1] > 0)[0]
        if not len(close):
            return None, {"missing": "no_close_command"}
        c = int(close[0])
        ks = [k for k in range(c - 5, c) if k >= 0]
        names = [str(n) for n in z["joint_names"]]
        a0 = qpos_addr(names)[target + "_joint0"][0]
        out = []
        for k in ks:
            d = float(np.linalg.norm(z["eef_pos"][k] - z["qpos"][k, a0:a0 + 3])) * 100
            st = "main" if d < 7.5 else ("mid" if d <= 15 else "far")
            out.append((k, lab[k], d, st))
        return out, {"t_c": c, "n": len(ks)}
    if rule == "preclose":
        T = len(lab)
        close = np.nonzero(np.asarray(z["action"])[:, -1] > 0)[0]
        c = int(close[0]) if len(close) else T
        b = [k for k in range(0, T, 8) if k < c][-(n_app + n_pre):]
        if len(b) < n_app + n_pre:
            return None, {"n_bounds_before_close": len(b)}
        return [(k, "pregrasp" if lab[k] == "pre-grasp" else lab[k]) for k in b], {"n_bounds_before_close": len(b), "first_close": c}
    T = len(lab)
    bounds = list(range(0, T, 8))
    pre = [k for k in bounds if lab[k] == "pre-grasp"]
    if len(pre) < n_pre:
        return None, {"n_pre_bounds": len(pre)}
    first_pre = min(k for k in range(T) if lab[k] == "pre-grasp")
    app = [k for k in bounds if lab[k] == "approach" and k < first_pre][-n_app:]
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(pre) - 1, n_pre)))
    pg = [pre[i] for i in idx]
    frames = [(k, "approach") for k in app] + [(k, "pregrasp") for k in pg]
    return frames, {"n_pre_bounds": len(pre), "n_app": len(app)}


def episodes(runs, cond, per_task, success_only, tasks=range(10), eps=None):
    """{task: [(ep, file)]} ascending ep, all candidates (selection of per_task with valid frames is done by caller)."""
    out = {}
    for t in tasks:
        fs = []
        for f in glob.glob(os.path.join(runs, cond, f"t{t}_e*_i*.npz")):
            ep = int(os.path.basename(f).split("_")[1][1:])
            if eps is None or ep in eps:
                fs.append((ep, f))
        out[t] = sorted(fs)
    return out


def seg_ids(env, sim, cam):
    return env._render(sim, cam, [], depth=False, seg=True)[2]


def gripper_geoms(m):
    return [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")]


def project_raw(sim, cam, pts, res=256):
    """World points -> (row, col) in the RAW rendered array (sim.render / read_pixels: OpenGL, row 0 = bottom of the
    view), pinhole from the MuJoCo camera pose and fovy (camera looks along its -z, x right, y up)."""
    m, d = sim.model, sim.data
    c = m.camera_name2id(cam)
    p, R = d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3)
    f = 0.5 * res / np.tan(np.radians(m.cam_fovy[c]) / 2)
    out = []
    for x in np.asarray(pts, float).reshape(-1, 3):
        xc = R.T @ (x - p)
        out.append([res / 2 + f * xc[1] / -xc[2], res / 2 + f * xc[0] / -xc[2]])
    return np.asarray(out)


def mask_shift(m0, m1, rng=40):
    """Integer (drow, dcol) maximising the overlap of mask m1 with m0 shifted (robust to parts leaving the view)."""
    best, arg = -1, (0, 0)
    r0, c0 = np.nonzero(m0)
    if not len(r0):
        return None
    H, W = m0.shape
    for dr in range(-rng, rng + 1, 1):
        rr = r0 + dr
        ok_r = (rr >= 0) & (rr < H)
        for dc in range(-rng, rng + 1, 1):
            cc = c0 + dc
            ok = ok_r & (cc >= 0) & (cc < W)
            v = int(m1[rr[ok], cc[ok]].sum())
            if v > best:
                best, arg = v, (dr, dc)
    return [int(arg[0]), int(arg[1])]


def project(sim, cam, pts, res=256):
    """World points -> (row, col) pixel coordinates in the raw (un-flipped) rendered image, by robosuite's camera
    matrix (its convention matches sim.render output, checked in the smoke against the segmentation centroid)."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    M = get_camera_transform_matrix(sim, cam, res, res)
    return project_points_from_world_to_camera(np.asarray(pts, float).reshape(-1, 3), M, res, res).astype(float)
