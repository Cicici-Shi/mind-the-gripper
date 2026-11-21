"""phasegate-pi (2026-11-10, exploration tier): copy of pi/zoo_client_mi.py. Additions only, all inside `if PG`
(env PG_MODE set) branches; with PG_MODE unset the client is the maskinfo-pi client. PG and MI_CELL are exclusive.
BodyEnv reads PG_MODE / PG_WINDOW itself (embody/phasegate.py); this client only:
  * requires --blind --proto openpi --body Panda, DUMP_DIR, MI_MODEL_TAG, MI_NOISE_MASTER (same fixed-noise server
    pi/serve_policy_mi.py and the same noise key as maskinfo-pi, so cells of one block share the noise per key);
  * asserts the in-process GL_RENDERER of the physical, reference and (look) twin contexts against PG_GL_ALLOW
    (comma list, default H200), as oft/run_libero_eval_pg.py;
  * records the per-step trace without the hook and writes one npz per episode (_pg_dump): the OFT dump fields
    (success, task, ep, init, action, eef_*, gripper_qpos, qpos, names, ref poses, contacts, host, gpu, align_json,
    twin_* in look mode), env.pg_fields(), per-query chunk / noise seed / noise md5; errored episodes go to
    <OUTROOT>_errors/ like the maskinfo-pi dump;
  * jsonl record: rec["pg"] = mode / window / lift_m (the per-step gate, phase and close / lift steps reveal episode
    length or outcome, so under --blind they live only in the npz); contacts and breakdown first steps dropped as MI.

maskinfo-pi (2026-11-08): copy of $PROJ_ROOT/vla_zoo/next/zoo_client_next.py (sha256 64cabe3f...). Additions only,
all inside `if MI` (env MI_CELL set) branches, so with MI_CELL unset the client behaves as the original:
  * embody.maskinfo_pi.MaskInfoPiHook on the physical BodyEnv (cells C0..C8, cadence MI_QUERY_EVERY, lag
    MI_LAG_STEPS, width MI_OUTLINE_W, protect MI_PROTECT, P list MI_PSET, model prep MI_PREP=openpi); call points
    exactly as oft/run_libero_eval_mi.py: begin_episode + arm(0) before set_init_state, finish(obs, 0) after;
    arm(step) before the env.step whose observation feeds the next query, finish(obs, step) after every step;
    note_model_input(el) after building the request; the 8-d state actually sent is compared with the pre-edit
    state observables of the same observation moment (ruling 11).
  * per-step state record (_mi_state, copied from run_libero_eval_mi.py) and one npz per episode in DUMP_DIR
    (t{task}_e{ep}_i{init}.npz; fields of the frozen maskinfo dump + mi_cell, mi_block, per-query raw chunks,
    noise fingerprints, server config). SKIP_DONE=1 skips keys whose npz exists (env still reset).
  * every request carries noise_seed = int.from_bytes(sha256(f"{MASTER}|{model}|{suite}|{task}|{ep}|{q}")[:8],
    "big") % 2**63 (protocol_maskinfo_pi.md v2 2.4; MASTER = MI_NOISE_MASTER, model = MI_MODEL_TAG pi05 | pi0,
    suite spatial | object, q = query index in the episode) and mi_q = q; never the cell or the arrival order.
    The server (pi/serve_policy_mi.py) draws the noise from it and returns its md5; seed and md5 are stored.
  * --blind is required with MI_CELL: no outcome / length in log or jsonl, no video; MI_TPUT_FILE (optional)
    receives "unix_time cumulative_steps cumulative_errors" every 5 s for the aggregate load test only.

next (2026-11-06): copy of vla_zoo/common/zoo_client.py (frozen models1 client, sha256 e0182776...) for the next
model batch (pi0-FAST via openpi z-score server, SmolVLA via zoo_server + adapter_lerobot). Changes, all additive:
  * --blind: the record and the log carry NO outcome and nothing that reveals episode length (no success, steps,
    queries, wall_s, contacts.n_steps); throughput is recorded per control step instead (sim_ms_step_med,
    loop_ms_step_mean, infer_ms_*). For interface smoke runs before preregistration.
  * twin: rot_err_deg_q and miss_q per query frame (tools/PATCH_NEXT_zoo_client_rot.md, ruling section 4.3).
  * physics breakdown detector D online (thresholds of tools/physics_breakdown_scan.py, unchanged): D-A arm joint
    jump > pi/2 between consecutive control steps or beyond Panda limits by 0.1 rad; D-B Robotiq finger_joint outside
    [-0.5, 1.2] (loose [-1, 2] flagged too); plus the MuJoCo BADQACC warning counter of the physical MjData. Only
    first_step and booleans are written (no counts that reveal episode length).
  * libero_goal / libero_10 suites (as g10/zoo_client_g10.py; embody.original first if EMBODY_DIR provides them).
  * --dump-inputs DIR: pickle the exact policy request at query indices --dump-queries (load-test replay).
Everything else (init selection, wait steps, BodyEnv, fixtures, contact log, twin fields) is byte-for-byte the
models1 logic, so keys pair with models1 / published cells.

vla_zoo client (2026-11-03): copy of $PROJ_ROOT/pi05/pi05_body_eval.py, generalised to any policy server.

--proto openpi : unchanged pi0.5 element (flip 180, resize_with_pad, 8-d state) -> openpi serve_policy (pi0, pi0-FAST).
--proto raw    : sends the raw env observation (256x256 images exactly as rendered, i.e. NOT flipped; eef pos,
                 eef quat xyzw, gripper qpos, joint pos, prompt). The model-specific server (vla_zoo/common/zoo_server.py
                 + an adapter) does all preprocessing and returns actions in LIBERO env convention
                 (7-d: delta pos, delta axis-angle, gripper -1 open / +1 close).
Everything else (suite, init-state selection, wait steps, BodyEnv, records) is identical to pi05_body_eval.py, so
episodes pair key-by-key with the OFT / pi0.5 runs.

Original docstring:
pi0.5-LIBERO websocket client on the 10 ORIGINAL libero_spatial tasks.

Mirrors openpi examples/libero/main.py (preprocessing, 8-d state, replan every 5 steps)
but uses embody.original.OriginalSpatialSuite and the OpenVLA-OFT init-state selection
(np.random.RandomState(5478 + task_id).permutation(n)[episode_idx]) so episodes pair with OFT runs.

Run from the `oftpi` conda env while a pi05_libero policy server listens on --port.
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody")))
from embody.original import OriginalSpatialSuite  # noqa: E402
if os.environ.get("M1_FASTSYNC") == "1":  # models1 2026-11-04: bitwise-equivalent fast TWIN_CAM_SYNC snapshot (m1_fastsync.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import m1_fastsync  # noqa: E402
    m1_fastsync.patch()
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256          # openpi renders LIBERO at 256 (OFT also uses env_img_res=256)
MAX_STEPS = 220        # libero_spatial (same in OFT and openpi)
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,  # NUS-PATCH 2026-10-27 (block 3: --suite)
                   "libero_goal": 300, "libero_10": 520}  # next: as g10/zoo_client_g10.py (openpi examples/libero/main.py)
PANDA_LIM = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973],
                      [-0.0175, 3.7525], [-2.8973, 2.8973]])  # = tools/physics_breakdown_scan.py LIM
FIN_LO, FIN_HI = -0.5, 1.2  # detector D-B (physics_breakdown_scan defaults); loose [-1, 2]


class BreakdownD:
    """Detector D of tools/physics_breakdown_scan.py, evaluated online on the PHYSICAL sim every control step."""

    def __init__(self, penv):
        self.penv = penv
        sim = penv.sim
        self.ai = list(penv.robots[0]._ref_joint_pos_indexes)
        try:
            jid = sim.model.joint_name2id("gripper0_finger_joint")
            self.fi = int(sim.model.jnt_qposadr[jid])
        except Exception:
            self.fi = None  # Panda gripper: finger slide joints, D-B not defined
        self.prev, self.A, self.B, self.loose, self.mj = None, None, None, False, 0.0
        self.k, self.qacc0 = 0, self._qacc()

    def _qacc(self):
        try:
            import mujoco
            return int(self.penv.sim.data._data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
        except Exception:
            return -1

    def update(self):
        q = np.asarray(self.penv.sim.data.qpos, np.float64)
        qa = q[self.ai]
        if self.prev is not None:
            dq = float(np.abs(qa - self.prev).max())
            self.mj = max(self.mj, dq)
            if dq > np.pi / 2 and self.A is None:
                self.A = {"first_step": self.k, "jump": True}
        if self.A is None and ((qa < PANDA_LIM[:, 0] - 0.1) | (qa > PANDA_LIM[:, 1] + 0.1)).any():
            self.A = {"first_step": self.k, "jump": False}
        if self.fi is not None:
            f = float(q[self.fi])
            if self.B is None and (f < FIN_LO or f > FIN_HI):
                self.B = {"first_step": self.k}
            self.loose = self.loose or f < -1.0 or f > 2.0
        self.prev = qa.copy()
        self.k += 1

    def summary(self):
        q1 = self._qacc()
        return {"A": self.A, "B": self.B, "breakdown": bool(self.A or self.B), "B_loose": bool(self.loose),
                "max_jump_deg": round(float(np.degrees(self.mj)), 1), "finger_joint": self.fi is not None,
                "qacc_warn": (q1 - self.qacc0) if (q1 >= 0 and self.qacc0 >= 0) else None}
NUM_STEPS_WAIT = 10
MI = bool(os.environ.get("MI_CELL"))  # maskinfo-pi
PG = bool(os.environ.get("PG_MODE"))  # phasegate-pi
assert not (MI and PG), "MI_CELL and PG_MODE are exclusive"


def _mi_state(env, mr):
    """maskinfo (verbatim from oft/run_libero_eval_mi.py): object / grip-site positions and gripper-object contacts
    of the CURRENT sim state."""
    if mr is None or mr.ep is None or mr.cell is None:
        return None
    rs = mr.rs
    sim, m = rs.sim, rs.sim.model
    if "mi_obj_bodies" not in mr.ep:
        bodies = []
        for n in mr.ep["targets"]:
            o = rs.objects_dict.get(n) if hasattr(rs, "objects_dict") else None
            if o is None and hasattr(rs, "fixtures_dict"):
                o = rs.fixtures_dict.get(n)
            bodies.append(int(m.body_name2id(o.root_body)) if o is not None else -1)
        gr = rs.robots[0].gripper
        gr = gr.get("right", next(iter(gr.values()))) if isinstance(gr, dict) else gr
        imp = gr.important_geoms
        gid = lambda names: {int(m.geom_name2id(x)) for x in names}
        grip_all = {g for g in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0_")}
        tgt = mr.ep["targets"][0] if mr.ep["targets"] else None
        tgt_geoms = {int(g) for g, inst in rs.model.geom_ids_to_instances.items() if inst == tgt}
        inter_geoms = {int(g) for g in mr.ep["P_geoms"]}
        mr.ep["mi_obj_bodies"] = bodies
        robot_all = {g for g in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(("robot0_", "mount0_"))}
        mr.ep["mi_geoms"] = (gid(imp["left_fingerpad"]), gid(imp["right_fingerpad"]), grip_all, tgt_geoms, inter_geoms,
                             robot_all)
        mr.ep["mi_pad_names"] = [list(imp["left_fingerpad"]), list(imp["right_fingerpad"])]
    L, R, A, T, I, RB = mr.ep["mi_geoms"]
    cl = cr = ca = ci = cs = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        for a, b in ((g1, g2), (g2, g1)):
            if b in T:
                cl |= a in L
                cr |= a in R
                ca |= a in A
                cs |= (a not in A) and (a not in T) and (a not in RB)  # support: table, other objects, basket, ...
            if a in A and b in I:
                ci = True
    pos = [sim.data.body_xpos[b].copy() if b >= 0 else np.full(3, np.nan) for b in mr.ep["mi_obj_bodies"]]
    site = sim.data.site_xpos[mr.ep["grip_site"]].copy()
    return pos, site, (cl, cr, ca, ci, cs)


def noise_seed_of(task, ep, q):
    """protocol_maskinfo_pi.md v2 2.4 (written there verbatim)."""
    import hashlib
    master, model = int(os.environ["MI_NOISE_MASTER"]), os.environ["MI_MODEL_TAG"]
    suite = os.environ["MI_SUITE_TAG"]
    return int.from_bytes(hashlib.sha256(f"{master}|{model}|{suite}|{task}|{ep}|{q}".encode()).digest()[:8], "big") % 2**63


def _state8(eef_pos, eef_quat, gripper_qpos):
    """The 8-d openpi LIBERO state, exactly as make_element builds it."""
    return np.concatenate((eef_pos, quat2axisangle(eef_quat), gripper_qpos)).astype(np.float64)


def _gl_renderers(env, with_twin=False):
    """GL_RENDERER of the render contexts this process actually uses (physical + reference sims), read in-process
    after they exist (hopper_capacity_2026-11-08: the production path silently falls back to llvmpipe when the
    NVIDIA EGL device fails, and an external gl_renderer.py process cannot see that). phasegate: with_twin adds the
    look-mode twin context (key "twin", None without a twin)."""
    from OpenGL import GL
    out = {}
    ctxs = [("physical", env.sim), ("reference", getattr(getattr(env, "ref", None), "sim", None))]
    if with_twin:
        ctxs.append(("twin", getattr(getattr(env, "twin", None), "sim", None)))
    for name, sim in ctxs:
        ctx = getattr(sim, "_render_context_offscreen", None) if sim is not None else None
        if ctx is None:
            out[name] = None
            continue
        ctx.gl_ctx.make_current()
        out[name] = GL.glGetString(GL.GL_RENDERER).decode()
    return out


class _Tput:
    """maskinfo-pi load test only: cumulative executed steps / errors of this process, every 5 s, to MI_TPUT_FILE."""

    def __init__(self, path):
        import threading
        self.path, self.steps, self.errs = path, 0, 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.wait(5.0):
            self.flush()

    def flush(self):
        with open(self.path, "a") as f:
            f.write(f"{time.time():.3f} {self.steps} {self.errs}\n")


def quat2axisangle(quat):
    """robosuite transform_utils.quat2axisangle (xyzw), as copied in openpi's example."""
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_raw(obs, prompt, suite):
    return {
        "agentview": np.ascontiguousarray(obs["agentview_image"]),
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
        "eef_pos": np.asarray(obs["robot0_eef_pos"], np.float64),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], np.float64),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float64),
        "joint_pos": np.asarray(obs["robot0_joint_pos"], np.float64),
        "prompt": str(prompt), "suite": suite,
    }


def make_element(obs, prompt, resize):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float64)
    return {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state, "prompt": str(prompt)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--trials", type=int, default=2, help="episodes per task")
    p.add_argument("--ep-offset", type=int, default=0, help="first episode index (permutation position)")
    p.add_argument("--tasks", default="all", help="comma list of task ids or 'all'")
    p.add_argument("--replan", type=int, default=0, help="actions executed per query; 0 = server metadata 'replan' (else 5)")
    p.add_argument("--max-steps", type=int, default=0, help="override the suite step budget (baseline alignment only)")
    p.add_argument("--proto", choices=("openpi", "raw"), default="raw")
    p.add_argument("--model", default="", help="model name written into every record")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7, help="global seed, same as OFT set_seed_everywhere(7)")
    p.add_argument("--blind", action="store_true", help="next: no outcome / episode-length fields in record or log")
    p.add_argument("--dump-inputs", default="", help="next: dir, pickle the exact request at --dump-queries indices")
    p.add_argument("--suite", default="libero_spatial", choices=list(SUITE_MAX_STEPS),
                   help="NUS-PATCH 2026-10-27: libero_object uses embody.original.OriginalObjectSuite (embody_v3)")
    p.add_argument("--body", default="", help="Panda|IIWA|UR5e: run on another body via embody.BodyEnv (HIDE_ARM env var hides the arm)")
    p.add_argument("--out", required=True, help="jsonl, one line per episode (appended)")
    p.add_argument("--skip-done", action="store_true",
                   help="skip (task, episode) already in --out without error; the env is still reset so the "
                        "global np.random stream (fixture placement) advances exactly as in a full run")
    p.add_argument("--dump-obs", default="", help="dir: save the policy input of selected queries as .npz")
    p.add_argument("--dump-queries", default="0,10", help="query indices per episode to dump")
    args = p.parse_args()
    if MI:  # maskinfo-pi: blind, openpi protocol, physical Panda only
        assert args.blind, "MI_CELL requires --blind"
        assert args.proto == "openpi" and args.body == "Panda", (args.proto, args.body)
        assert os.environ.get("DUMP_DIR") and os.environ.get("MI_BLOCK"), "MI_CELL needs DUMP_DIR and MI_BLOCK"
        assert os.environ.get("MI_MODEL_TAG") in ("pi05", "pi0") and os.environ.get("MI_NOISE_MASTER"), "noise key"
        os.environ["MI_SUITE_TAG"] = args.suite.replace("libero_", "")
        assert os.environ["MI_SUITE_TAG"] in ("spatial", "object"), args.suite
        assert os.environ["MI_BLOCK"] == f"{os.environ['MI_MODEL_TAG']}_{os.environ['MI_SUITE_TAG']}", os.environ["MI_BLOCK"]
        os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
    if PG:  # phasegate-pi: blind, openpi protocol, physical Panda, fixed-noise server (same noise key as maskinfo-pi)
        assert args.blind, "PG_MODE requires --blind"
        assert args.proto == "openpi" and args.body == "Panda", (args.proto, args.body)
        assert os.environ.get("DUMP_DIR") and os.environ.get("PG_WINDOW"), "PG_MODE needs DUMP_DIR and PG_WINDOW"
        assert os.environ.get("MI_MODEL_TAG") in ("pi05", "pi0") and os.environ.get("MI_NOISE_MASTER"), "noise key"
        os.environ["MI_SUITE_TAG"] = args.suite.replace("libero_", "")
        assert os.environ["MI_SUITE_TAG"] in ("spatial", "object", "goal", "10"), args.suite  # hmz_breadth: + goal, 10
        os.makedirs(os.environ["DUMP_DIR"], exist_ok=True)
    tput =_Tput(os.environ["MI_TPUT_FILE"]) if os.environ.get("MI_TPUT_FILE") else None

    np.random.seed(args.seed)
    random.seed(args.seed)
    global MAX_STEPS
    MAX_STEPS = args.max_steps or SUITE_MAX_STEPS[args.suite]
    if args.suite in ("libero_goal", "libero_10"):  # next: embody.original if EMBODY_DIR has it, else g10 rebuild
        import embody.original as _eo
        _cls = {"libero_goal": "OriginalGoalSuite", "libero_10": "Original10Suite"}[args.suite]
        if hasattr(_eo, _cls):
            suite = getattr(_eo, _cls)()
        else:
            sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/vla_zoo/g10"))
            from original_g10 import SUITES as _G10
            suite = _G10[args.suite]()
        print(f"suite {args.suite} from {type(suite).__module__}.{type(suite).__name__}", flush=True)
    elif args.suite == "libero_object":
        from embody.original import OriginalObjectSuite
        suite = OriginalObjectSuite()
    else:
        suite = OriginalSpatialSuite()
    try:  # NUS-PATCH 2026-10-27: online arm-environment contacts (embody_v3 only; older EMBODY_DIR has none)
        from embody import contacts as _contacts
    except ImportError:
        _contacts = None
    task_ids = range(suite.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    dump_q = {int(x) for x in args.dump_queries.split(",") if x != ""}
    if args.dump_obs:
        os.makedirs(args.dump_obs, exist_ok=True)
    if args.dump_inputs:
        os.makedirs(args.dump_inputs, exist_ok=True)
        import pickle

    # The server compiles on its first inference (tens of seconds on A100) and blocks its event loop
    # meanwhile, so the client's 20 s keepalive ping times out and closes the connection; every later
    # episode then fails with ConnectionClosedError and is recorded as a policy failure. Localhost
    # needs no keepalive: disable it, and reconnect + retry the query if the connection still drops.
    import websockets.sync.client as _wsc
    import websockets.exceptions as _wse
    _connect = _wsc.connect
    _wsc.connect = lambda *a, **k: _connect(*a, **{"ping_interval": None, **k})
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata()
    print("server metadata:", meta, flush=True)
    if not args.replan:
        args.replan = int((meta or {}).get("replan", 5))
    print(f"replan={args.replan} proto={args.proto} model={args.model}", flush=True)
    if MI:  # maskinfo-pi: effective configuration (gate item 7); the server publishes its own in the metadata
        srv_cfg = (meta or {}).get("mi_server_cfg")
        assert srv_cfg is not None, "server is not pi/serve_policy_mi.py (no mi_server_cfg in metadata)"
        assert args.replan == int(os.environ.get("MI_QUERY_EVERY", "5")), (args.replan, os.environ.get("MI_QUERY_EVERY"))
        print("MI_CFG " + json.dumps({"model": args.model, "suite": args.suite, "replan": args.replan,
                                      "block": os.environ.get("MI_BLOCK"), "cell": os.environ.get("MI_CELL"),
                                      "query_every": os.environ.get("MI_QUERY_EVERY"),
                                      "lag": os.environ.get("MI_LAG_STEPS"), "width": os.environ.get("MI_OUTLINE_W"),
                                      "protect": os.environ.get("MI_PROTECT"), "prep": os.environ.get("MI_PREP", "openpi"),
                                      "embody_dir": os.environ.get("EMBODY_DIR"), "server": srv_cfg}), flush=True)
    if PG:  # phasegate-pi: the server must be the fixed-noise one (it refuses requests without noise_seed anyway)
        srv_cfg = (meta or {}).get("mi_server_cfg")
        assert srv_cfg is not None, "server is not pi/serve_policy_mi.py (no mi_server_cfg in metadata)"
        print("PG_CFG " + json.dumps({"model": args.model, "suite": args.suite, "replan": args.replan,
                                      "mode": os.environ.get("PG_MODE"), "window": os.environ.get("PG_WINDOW"),
                                      "lift_m": os.environ.get("PG_LIFT_M", "0.03"),
                                      "twin": {k: os.environ.get(k) for k in ("TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT")},
                                      "noise_master": os.environ.get("MI_NOISE_MASTER"),
                                      "gl_allow": os.environ.get("PG_GL_ALLOW", "H200"),
                                      "embody_dir": os.environ.get("EMBODY_DIR"), "server": srv_cfg,
                                      **({"yoke_dir": os.environ["PG_YOKE_DIR"]} if os.environ.get("PG_YOKE_DIR")
                                         else {})}), flush=True)

    def infer(el):
        nonlocal client
        for attempt in range(4):
            try:
                return client.infer(el)
            except (_wse.ConnectionClosed, ConnectionError, OSError) as e:
                print(f"INFER_RETRY attempt={attempt} {e!r}", flush=True)
                time.sleep(5)
                client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        return client.infer(el)

    done_keys = set()
    if args.skip_done and os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done_keys.add((r["task"], r["episode"]))

    import socket  # NUS-PATCH 2026-10-31 (block 3 amendment 1, R8/R10): node and GPU model in every record
    host, gpu = socket.gethostname(), os.environ.get("NUS_GPU", "")
    if not gpu:
        try:
            import subprocess
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=30).stdout.strip().splitlines()[0]
        except Exception as e:
            gpu = f"unknown ({e!r})"
    n_ep = n_succ = 0
    gl_seen = {}
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(0)
        if args.body:  # same wrapper as the OFT runs: Panda reference does the wait steps, scene transferred
            from embody.body_env import BodyEnv
            env = BodyEnv(env, bddl, args.body, ENV_RES, NUM_STEPS_WAIT, log=lambda m: print(m, flush=True))
        mr = None
        if MI:  # maskinfo-pi hook on the physical env (as run_libero_eval_mi.py run_task)
            from embody.maskinfo_pi import MaskInfoPiHook
            assert getattr(env, "skip_wait", False), "BodyEnv must do the wait steps on the reference (step 0 = query 0)"
            mr = MaskInfoPiHook(env, os.environ["MI_CELL"], res=ENV_RES, metrics_path=os.environ.get("MR_METRICS"),
                                log=lambda m: print(m, flush=True), lag_steps=int(os.environ["MI_LAG_STEPS"]),
                                outline_width=int(os.environ["MI_OUTLINE_W"]), protect_set=os.environ["MI_PROTECT"],
                                query_every=int(os.environ["MI_QUERY_EVERY"]), prep=os.environ.get("MI_PREP", "openpi"))
            print(f"MASKINFO: cell={mr.cell} spec={json.dumps(mr.spec)}", flush=True)
        perm = np.random.RandomState(5478 + task_id).permutation(len(init_states))
        for ep in range(args.ep_offset, args.ep_offset + args.trials):
            init_id = int(perm[ep])
            t0 = time.time()
            env.reset()
            if (task_id, ep) in done_keys:
                continue
            if (MI or PG) and os.environ.get("SKIP_DONE") == "1" and glob.glob(
                    os.path.join(os.environ["DUMP_DIR"], f"t{task_id}_e{ep}_i*.npz")):
                print(f"SKIP_DONE task={task_id} ep={ep}", flush=True)
                continue
            if mr is not None:
                mr.begin_episode(task_id, ep)
                mr.arm(0)
            if PG and os.environ.get("PG_YOKE_DIR"):  # phasegate_yoke: fixed window from the same key in the source cell
                _yc, _yl, _ys = _pg_yoke_src(os.environ["PG_YOKE_DIR"], task_id, ep, init_id, args)
                env.pg_set_yoke(_yc, _yl, _ys)
                print(f"PG_YOKE task={task_id} ep={ep} src={os.path.basename(_ys)}", flush=True)  # no times (blind log)
            obs = env.set_init_state(init_states[init_id])
            mi_rec, mi_q, trace, mi_s_server = [], [], None, None
            if mr is not None and gl_seen.get(id(env)) is None:  # in-process GL_RENDERER assert, once per env
                glr = _gl_renderers(env)
                want = os.environ.get("MI_GL_EXPECT", "H200")
                print("MI_GL_RENDERER " + json.dumps(glr), flush=True)
                if glr.get("physical") is None or any(v is not None and (want not in v or "llvmpipe" in v.lower())
                                                      for v in glr.values()):
                    print(f"ASSERT GL_RENDERER in-process {glr} (want {want}, no llvmpipe)", flush=True)
                    sys.exit(6)
                gl_seen[id(env)] = glr
                env._mi_glr = glr
            if PG and gl_seen.get(id(env)) is None:  # phasegate-pi: same assert, allow-list, twin context included
                glr = _gl_renderers(env, with_twin=True)
                allow = os.environ.get("PG_GL_ALLOW", "H200").split(",")
                print("PG_GL_RENDERER " + json.dumps(glr), flush=True)
                if glr.get("physical") is None or (os.environ.get("PG_MODE") == "look" and glr.get("twin") is None) or \
                        any(v is not None and (not any(a in v for a in allow) or "llvmpipe" in v.lower())
                            for v in glr.values()):
                    print(f"ASSERT GL_RENDERER in-process {glr} (allow {allow}, no llvmpipe)", flush=True)
                    sys.exit(6)
                gl_seen[id(env)] = glr
                env._mi_glr = glr
            if PG:  # phasegate-pi: per-step trace without the hook (same fields as the OFT dump)
                _m, _rm = env.sim.model, env.ref.sim.model
                trace = {"action": [], "eef_pos": [], "eef_quat": [], "gripper_qpos": [], "qpos": [],
                         "joint_names": [_m.joint_id2name(j) or "" for j in range(_m.njnt)],
                         "ref_body_names": [_rm.body_id2name(b) or "" for b in range(_rm.nbody)],
                         "ref_body_pos": _rm.body_pos.copy(), "ref_body_quat": _rm.body_quat.copy()}
            if mr is not None:
                mr.finish(obs, 0)
                _st = _mi_state(env, mr)
                if _st is not None:
                    mi_rec.append(_st)
                _m, _rm = env.sim.model, env.ref.sim.model
                trace = {"action": [], "eef_pos": [], "eef_quat": [], "gripper_qpos": [], "qpos": [],
                         "joint_names": [_m.joint_id2name(j) or "" for j in range(_m.njnt)],
                         "ref_body_names": [_rm.body_id2name(b) or "" for b in range(_rm.nbody)],
                         "ref_body_pos": _rm.body_pos.copy(), "ref_body_quat": _rm.body_quat.copy()}
            # scene fingerprint: fixture positions of the Panda reference scene, compared with the OFT
            # dumps (ref_body_names/pos) to confirm both models saw the same scene for this (task, episode)
            ref_env = env.ref if args.body else env
            from embody import align as _align
            fix = {k: [round(float(x), 5) for x in v[0]] for k, v in _align.static_body_pose(ref_env).items()}
            objq = {k: [round(float(x), 5) for x in v[0]] for k, v in _align.object_state_by_name(ref_env).items()}
            clog, clog_err = None, None
            plan = collections.deque()
            t = NUM_STEPS_WAIT if getattr(env, "skip_wait", False) else 0
            n_q, infer_ms, success, err = 0, [], False, None
            a0, chunk_len = None, None  # models1 (2026-11-04): first executed block (E1/E2 bitwise check), chunk length
            penv = env.env if args.body else env  # next: physical env (BodyEnv target body, or the native env)
            det = BreakdownD(penv)
            sim_ms, srv_ms, t_loop0, n_exec = [], [], None, 0
            try:
                while t < MAX_STEPS + NUM_STEPS_WAIT:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    if not plan:
                        el = (make_element(obs, task.language, args.resize) if args.proto == "openpi"
                              else make_raw(obs, task.language, args.suite))
                        if args.proto == "raw":
                            el["new_episode"] = n_q == 0
                        if args.dump_obs and n_q in dump_q and args.proto == "openpi":
                            np.savez_compressed(
                                os.path.join(args.dump_obs, f"t{task_id}_e{ep}_q{n_q}.npz"),
                                image=el["observation/image"], wrist_image=el["observation/wrist_image"],
                                state=el["observation/state"], prompt=el["prompt"],
                            )
                        if args.dump_inputs and n_q in dump_q:
                            with open(os.path.join(args.dump_inputs, f"t{task_id}_e{ep}_q{n_q}.pkl"), "wb") as fh:
                                pickle.dump({k: v for k, v in el.items()}, fh)
                        if mr is not None:  # maskinfo-pi: model input md5 + same-moment state check + noise key
                            mr.note_model_input(el)
                            pre = mr.pre_edit_state()  # this frame's (None if not captured)
                            s_ok = pre is not None and np.array_equal(
                                el["observation/state"], _state8(pre["robot0_eef_pos"], pre["robot0_eef_quat"],
                                                                 pre["robot0_gripper_qpos"]))
                            mr.note_state(s_ok)
                            el["noise_seed"] = noise_seed_of(int(task_id), int(ep), int(n_q))
                            el["mi_q"] = int(n_q)
                        if PG:  # phasegate-pi: same noise key as maskinfo-pi (no cell in it: common across cells)
                            el["noise_seed"] = noise_seed_of(int(task_id), int(ep), int(n_q))
                            el["mi_q"] = int(n_q)
                        if t_loop0 is None:
                            t_loop0 = time.time()
                        tq = time.time()
                        res = infer(el)
                        if mr is not None:
                            nz = res.get("mi_noise") if isinstance(res, dict) else None
                            assert nz is not None and int(nz["q"]) == n_q and int(nz["seed"]) == el["noise_seed"], \
                                f"server returned no / wrong noise fingerprint {nz}"
                            if n_q == 0:  # server-side record of the first request's state (raw + transformed)
                                ss = res.get("mi_S_server")
                                assert ss is not None, "server did not record the first request's state"
                                mi_s_server = {"received": np.asarray(ss["received"], np.float64),
                                               "transformed": None if ss.get("transformed") is None
                                               else np.asarray(ss["transformed"], np.float32),
                                               "eq_sent": bool(np.array_equal(np.asarray(ss["received"], np.float64),
                                                                              el["observation/state"]))}
                            mi_q.append({"step": len(trace["action"]), "noise_md5": str(nz["md5"]),
                                         "noise_seed": int(el["noise_seed"]), "S_ok": bool(s_ok),
                                         "chunk": np.asarray(res["actions"], np.float32)})
                        if PG:
                            nz = res.get("mi_noise") if isinstance(res, dict) else None
                            assert nz is not None and int(nz["q"]) == n_q and int(nz["seed"]) == el["noise_seed"], \
                                f"server returned no / wrong noise fingerprint {nz}"
                            mi_q.append({"step": len(trace["action"]), "noise_md5": str(nz["md5"]),
                                         "noise_seed": int(el["noise_seed"]),
                                         "chunk": np.asarray(res["actions"], np.float32)})
                        infer_ms.append((time.time() - tq) * 1000)
                        st_ = res.get("server_timing") if isinstance(res, dict) else None
                        if st_ and "infer_ms" in st_:
                            srv_ms.append(float(st_["infer_ms"]))
                        chunk = res["actions"]
                        chunk = np.asarray(chunk, np.float64).reshape(-1, 7)
                        assert len(chunk) >= args.replan, chunk.shape
                        if a0 is None:
                            a0, chunk_len = chunk[: args.replan].tolist(), int(len(chunk))
                        plan.extend(chunk[: args.replan])
                        n_q += 1
                    action = plan.popleft()
                    if clog is None and _contacts is not None and clog_err is None:
                        try:
                            clog = _contacts.make(env)
                        except Exception as ce:  # never let the contact log break an episode
                            clog_err = repr(ce)
                    if mr is not None:  # maskinfo-pi: trace before the step (as run_libero_eval_mi.py)
                        trace["action"].append(np.asarray(action, dtype=np.float32))
                        trace["eef_pos"].append(obs["robot0_eef_pos"].copy())
                        trace["eef_quat"].append(obs["robot0_eef_quat"].copy())
                        trace["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
                        trace["qpos"].append(env.sim.data.qpos.copy())
                        if not plan:
                            mr.arm(len(trace["action"]))  # this step's observation feeds the next query
                    if PG:  # phasegate-pi: trace before the step (pre-action observation, as the OFT dump)
                        trace["action"].append(np.asarray(action, dtype=np.float32))
                        trace["eef_pos"].append(obs["robot0_eef_pos"].copy())
                        trace["eef_quat"].append(obs["robot0_eef_quat"].copy())
                        trace["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
                        trace["qpos"].append(env.sim.data.qpos.copy())
                    ts_ = time.time()
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
                    sim_ms.append((time.time() - ts_) * 1000)
                    n_exec += 1
                    if tput is not None:
                        tput.steps += 1
                    if mr is not None:
                        mr.finish(obs, len(trace["action"]))
                        _st = _mi_state(env, mr)
                        if _st is not None:
                            mi_rec.append(_st)
                    try:
                        det.update()
                    except Exception as de:
                        det.A = det.A or {"error": repr(de)}
                    if clog is not None:
                        try:
                            clog.update(np.asarray(action))
                        except Exception as ce:
                            clog_err, clog = repr(ce), None
                    if done:
                        success = True
                        break
                    t += 1
            except Exception as e:  # keep going; record the error
                import traceback
                err = f"{e!r}"
                traceback.print_exc()
                if mr is not None:  # review 11-08 item 2: never leave the hook armed into the next reset
                    mr.armed, mr.cap, mr._next_step = False, None, None
            if tput is not None and err is not None:
                tput.errs += 1
            if mr is not None:  # maskinfo-pi: npz dump (never printed; outcome only inside the npz)
                try:
                    _mi_dump(env, mr, trace, mi_rec, mi_q, success, err, task_id, ep, init_id, host, gpu, args,
                             chunk_len, srv_cfg, clog, clog_err, mi_s_server)
                    mr.end_episode()
                except Exception as de:
                    import traceback
                    traceback.print_exc()
                    print(f"DUMP_ERROR task={task_id} ep={ep} {de!r}", flush=True)
                    if tput is not None:
                        tput.errs += 1
            if PG:  # phasegate-pi: npz dump (outcome only inside the npz)
                try:
                    _pg_dump(env, trace, mi_q, success, err, task_id, ep, init_id, host, gpu, args, chunk_len, srv_cfg,
                             clog, clog_err)
                except Exception as de:
                    import traceback
                    traceback.print_exc()
                    print(f"DUMP_ERROR task={task_id} ep={ep} {de!r}", flush=True)
            n_ep += 1
            n_succ += int(success)
            rec = {
                "model": args.model, "proto": args.proto, "max_steps": MAX_STEPS, "replan": args.replan, "task": task_id, "task_name": task.name, "episode": ep, "init": init_id, "success": success,
                "steps": t - NUM_STEPS_WAIT, "queries": n_q, "wall_s": round(time.time() - t0, 2),
                "suite": args.suite, "objq": objq, "host": host, "gpu": gpu,
                "align": getattr(env, "align_info", None),
                "contacts": clog.summary() if clog is not None else ({"error": clog_err} if clog_err else None),
                "a0": a0, "chunk_len": chunk_len, "server_host": os.environ.get("NUS_SRV_HOST", args.host),
                "server_gpu": os.environ.get("NUS_SRV_GPU", ""), "embody_dir": os.environ.get("EMBODY_DIR", ""), "fastsync": os.environ.get("M1_FASTSYNC", "0"),
                "rtt_ms_med": round(float(np.median(infer_ms)), 1) if infer_ms else None,
                "cell_env": {k: os.environ.get(k) for k in ("BODY", "GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER",
                             "TWIN_BODY", "TWIN_CAMS", "TWIN_CAM", "TWIN_CAM_SYNC") if os.environ.get(k) is not None},
                "body": args.body or "SRC", "hide_arm": os.environ.get("HIDE_ARM", "0"), "rtt_ms_mean": round(float(np.mean(infer_ms)), 1) if infer_ms else None, "error": err, "fix": fix,
                "twin": None if getattr(env, "twin_stats", None) is None else {
                    "pos_err_mm_max": float(np.max(env.twin_stats["pos_err_mm"])),
                    "rot_err_deg_max": float(np.max(env.twin_stats["rot_err_deg"])),
                    "n_paint": len(env.twin_stats["pos_err_mm"]),
                    "miss": int(sum(1 for a, b in zip(env.twin_stats["pos_err_mm"], env.twin_stats["rot_err_deg"]) if a > 2 or b > 1)),
                    "posture_deg_p95": float(np.percentile(env.twin_stats["posture_deg"], 95)) if env.twin_stats["posture_deg"] else None,
                    "stale": env.twin_stats["stale"], "t0": env.twin_stats["t0"],
                    "over_1mm": int(sum(1 for a in env.twin_stats["pos_err_mm"] if a > 1)),
                    "over_1mm_q": int(sum(1 for a in env.twin_stats["pos_err_mm"][::args.replan] if a > 1)),
                    **({} if not env.twin_stats.get("cam_err_mm") else {  # models1: twin vs physical wrist camera
                        "cam_err_mm_max": float(np.max(env.twin_stats["cam_err_mm"])),
                        "cam_err_deg_max": float(np.max(env.twin_stats["cam_err_deg"])),
                        "cam_err_mm_mean": float(np.mean(env.twin_stats["cam_err_mm"])),
                        "cam_err_mm_q": [round(float(a), 5) for a in env.twin_stats["cam_err_mm"][::args.replan]],
                        "cam_err_deg_q": [round(float(a), 5) for a in env.twin_stats["cam_err_deg"][::args.replan]]}),
                    "pos_err_mm_q": [round(float(a), 4) for a in env.twin_stats["pos_err_mm"][::args.replan]],
                    **({} if not env.twin_stats.get("root_off_mm") else {  # NUS-PATCH 2026-10-31: G_R root offset
                        "root_off_mm_q": [[round(float(x), 3) for x in v] for v in env.twin_stats["root_off_mm"][::args.replan]]}),
                },
            }
            # next: throughput per control step, physics breakdown detector D, rotation residual per query frame
            loop_s = (time.time() - t_loop0) if t_loop0 is not None else None
            rec["timing"] = {"sim_ms_step_med": round(float(np.median(sim_ms)), 2) if sim_ms else None,
                             "loop_ms_step_mean": round(1000 * loop_s / n_exec, 2) if (loop_s and n_exec) else None,
                             "infer_ms_med": rec["rtt_ms_med"], "server_infer_ms_med":
                             round(float(np.median(srv_ms)), 1) if srv_ms else None, "replan": args.replan}
            rec["breakdown_D"] = det.summary()
            if MI:  # review 11-08 item 4: contact summary / breakdown first steps are outcome / length proxies
                rec.pop("contacts", None)
                rec["breakdown_D"] = {k: v for k, v in rec["breakdown_D"].items() if k not in ("A", "B")}
            if PG:  # phasegate-pi: same proxies dropped; only the cell configuration of pg_fields in the record
                rec.pop("contacts", None)
                rec["breakdown_D"] = {k: v for k, v in rec["breakdown_D"].items() if k not in ("A", "B")}
                _pf = env.pg_fields()
                rec["pg"] = {"mode": str(_pf["pg_mode"]), "window": str(_pf["pg_window"]),
                             "lift_m": float(_pf["pg_lift_m"])}
                if os.environ.get("PG_YOKE_DIR"):  # phasegate_yoke: source cell only (the times stay in the npz)
                    rec["pg"]["yoke_dir"] = os.environ["PG_YOKE_DIR"]
            if rec["twin"] is not None:
                _pe = env.twin_stats["pos_err_mm"][::args.replan]
                _re = env.twin_stats["rot_err_deg"][::args.replan]
                rec["twin"]["rot_err_deg_q"] = [round(float(a), 4) for a in _re]
                rec["twin"]["miss_q"] = int(sum(1 for a, b in zip(_pe, _re) if a > 2 or b > 1))
            if args.blind:  # next: nothing that is, or reveals, the outcome or the episode length
                for k_ in ("success", "steps", "queries", "wall_s"):
                    rec.pop(k_, None)
                if isinstance(rec.get("contacts"), dict):
                    rec["contacts"].pop("n_steps", None)
                if isinstance(rec.get("twin"), dict):
                    rec["twin"].pop("n_paint", None)  # = episode length
                    if PG:  # phasegate-pi (look): per-query lists have one entry per query = length proxy (npz has them)
                        for k_ in ("pos_err_mm_q", "rot_err_deg_q", "cam_err_mm_q", "cam_err_deg_q", "root_off_mm_q"):
                            rec["twin"].pop(k_, None)
                rec["blind"] = True
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            # models1 7.5: no outcome in the log (only interface quantities); next --blind: no length proxies either
            if args.blind:
                print(f"EP task={task_id} ep={ep} init={init_id} error={err}", flush=True)
            else:
                print(f"EP task={task_id} ep={ep} init={init_id} queries={n_q} wall_s={rec['wall_s']} error={err}", flush=True)
        env.close()
    if tput is not None:
        tput.flush()
    print(f"FINAL episodes {n_ep}", flush=True)  # models1 7.5: success count no longer printed


def _mi_dump(env, mr, trace, mi_rec, mi_q, success, err, task_id, ep, init_id, host, gpu, args, chunk_len, srv_cfg,
             clog, clog_err, mi_s_server=None):
    """maskinfo-pi npz: the frozen maskinfo dump fields (run_libero_eval_mi.py _dump_episode + mi_extra) plus the
    openpi additions. File name t{task}_e{ep}_i{init}.npz in DUMP_DIR."""
    d = os.environ["DUMP_DIR"]
    if not trace or not trace["action"]:
        print(f"DUMP_SKIPPED_EMPTY task={task_id} ep={ep} error={err}", flush=True)
        return
    contacts = clog.npz_fields() if clog is not None else ({"contact_error": clog_err} if clog_err else {})
    mi_extra = {}
    if mi_rec:
        mi_extra = dict(mi_obj_names=np.asarray(mr.ep["targets"]), mi_obj_pos=np.asarray([r[0] for r in mi_rec], np.float64),
                        mi_grip_pos=np.asarray([r[1] for r in mi_rec], np.float64),
                        mi_contact_lpad=np.asarray([r[2][0] for r in mi_rec], bool),
                        mi_contact_rpad=np.asarray([r[2][1] for r in mi_rec], bool),
                        mi_contact_grip_target=np.asarray([r[2][2] for r in mi_rec], bool),
                        mi_contact_grip_interactive=np.asarray([r[2][3] for r in mi_rec], bool),
                        mi_contact_target_support=np.asarray([r[2][4] for r in mi_rec], bool),
                        mi_pad_geoms=json.dumps(mr.ep["mi_pad_names"]),
                        mi_block=os.environ.get("MI_BLOCK", ""))
    hook = mr.dump_fields()
    assert hook.get("mi_cell") == os.environ["MI_CELL"], (hook.get("mi_cell"), os.environ["MI_CELL"])
    hook["mi_cell"] = os.environ["MI_CELL"]  # protocol v2 4 item 10: written explicitly by the client
    # review 11-08 item 3: an errored episode goes to <OUTROOT>_errors/ (so SKIP_DONE reruns it and the readout never
    # sees it); every npz is written to a temp name and renamed, so a killed job leaves no half-written npz.
    if err is not None:  # <OUTROOT>/<block>/<cell> -> <OUTROOT>_errors/<block>/<cell>
        dd = os.path.normpath(d)
        d = os.path.join(os.path.dirname(os.path.dirname(dd)) + "_errors", *dd.split(os.sep)[-2:])
        os.makedirs(d, exist_ok=True)
    final = os.path.join(d, f"t{task_id}_e{ep}_i{init_id}.npz")
    tmp = os.path.join(d, f".tmp_{os.getpid()}_t{task_id}_e{ep}_i{init_id}.npz")  # never matches t*_e*_i*.npz
    np.savez_compressed(
        tmp,
        success=success, task=task_id, ep=ep, init=init_id,
        action=np.asarray(trace["action"], dtype=np.float32),
        eef_pos=np.asarray(trace["eef_pos"], dtype=np.float32),
        eef_quat=np.asarray(trace["eef_quat"], dtype=np.float32),
        gripper_qpos=np.asarray(trace["gripper_qpos"], dtype=np.float32),
        qpos=np.asarray(trace["qpos"], dtype=np.float64),
        joint_names=np.asarray(trace["joint_names"]),
        ref_body_names=np.asarray(trace["ref_body_names"]),
        ref_body_pos=trace["ref_body_pos"], ref_body_quat=trace["ref_body_quat"],
        **contacts,
        host=host, gpu=gpu, align_json=json.dumps(getattr(env, "align_info", None)),
        **hook, **mi_extra,
        query_step=np.asarray([q["step"] for q in mi_q], np.int32),
        query_chunk=np.asarray([q["chunk"] for q in mi_q], np.float32),
        mi_noise_md5=np.asarray([q["noise_md5"] for q in mi_q]),
        noise_seed=np.asarray([q["noise_seed"] for q in mi_q], np.int64),
        mi_noise_master=np.int64(int(os.environ["MI_NOISE_MASTER"])), mi_model_tag=os.environ["MI_MODEL_TAG"],
        mi_S_sent_ok=np.asarray([q["S_ok"] for q in mi_q], bool),
        mi_model=args.model, mi_suite=args.suite, mi_replan=np.int32(args.replan),
        mi_chunk_len=np.int32(chunk_len if chunk_len is not None else -1),
        mi_server_cfg=json.dumps(srv_cfg), mi_error=json.dumps(err),
        server_host=os.environ.get("NUS_SRV_HOST", args.host), server_gpu=os.environ.get("NUS_SRV_GPU", ""),
        embody_dir=os.environ.get("EMBODY_DIR", ""), mi_blind=True,
        mi_gl_renderer=json.dumps(getattr(env, "_mi_glr", None)),
        **({} if mi_s_server is None else dict(mi_S_server_received=mi_s_server["received"],  # noqa
                                               **({} if mi_s_server["transformed"] is None else
                                                  {"mi_S_server_transformed": mi_s_server["transformed"]}),
                                               mi_S_server_eq_sent=mi_s_server["eq_sent"])),
    )
    os.replace(tmp, final)


def _pg_yoke_src(ydir, task_id, ep, init_id, args):
    """phasegate_yoke: (t_close, t_lift, path) of the same key (task, ep, init) in the source cell. Reads only the
    event times and the configuration fields; fails loudly (no fallback to the event window) if the file is missing
    or the source is not an unintervened "none" cell of the same model, suite and lift threshold."""
    f = os.path.join(os.path.abspath(os.path.expanduser(ydir)), f"t{task_id}_e{ep}_i{init_id}.npz")
    assert os.path.isfile(f), f"PG_YOKE_DIR has no source episode for task={task_id} ep={ep} init={init_id}: {f}"
    with np.load(f, allow_pickle=False) as z:
        assert str(z["pg_window"]) == "none", f"yoke source is not a none cell: {f} window={z['pg_window']}"
        assert "pg_yoke" not in z.files, f"yoke source is itself yoked: {f}"
        assert (int(z["task"]), int(z["ep"]), int(z["init"])) == (task_id, ep, init_id), f
        assert str(z["mi_model_tag"]) == os.environ["MI_MODEL_TAG"] and str(z["pg_suite"]) == args.suite, \
            (f, str(z["mi_model_tag"]), str(z["pg_suite"]))
        assert float(z["pg_lift_m"]) == float(os.environ.get("PG_LIFT_M", "0.03")), (f, float(z["pg_lift_m"]))
        return int(z["pg_t_close"]), int(z["pg_t_lift"]), f


def _pg_dump(env, trace, pg_q, success, err, task_id, ep, init_id, host, gpu, args, chunk_len, srv_cfg, clog, clog_err):
    """phasegate-pi npz: the OFT dump fields (oft/run_libero_eval_pg.py _dump_episode, incl. twin_* in look mode) plus
    env.pg_fields() and the openpi query / noise records. Same file name, temp-then-rename and error routing as
    _mi_dump, so analysis/pg_readout.py reads OFT and pi cells alike."""
    d = os.environ["DUMP_DIR"]
    if not trace or not trace["action"]:
        print(f"DUMP_SKIPPED_EMPTY task={task_id} ep={ep} error={err}", flush=True)
        return
    pg = env.pg_fields()
    assert pg and str(pg["pg_mode"]) == os.environ["PG_MODE"] and str(pg["pg_window"]) == os.environ["PG_WINDOW"], pg
    contacts = clog.npz_fields() if clog is not None else ({"contact_error": clog_err} if clog_err else {})
    tw = getattr(env, "twin_stats", None)
    twin = {}
    if tw:  # as the OFT dump (TWIN_RENDER checks, embody BodyEnv.twin_stats)
        twin = dict(twin_pos_err_mm=np.asarray(tw["pos_err_mm"], dtype=np.float32),
                    twin_rot_err_deg=np.asarray(tw["rot_err_deg"], dtype=np.float32),
                    twin_posture_deg=np.asarray(tw["posture_deg"], dtype=np.float32),
                    twin_stale=tw["stale"],
                    twin_t0=np.asarray([v for k in ("agentview_image", "robot0_eye_in_hand_image") for v in tw["t0"][k]],
                                       dtype=np.float32) if tw.get("t0") else np.zeros(4, np.float32))
        if tw.get("root_off_mm"):
            twin["twin_root_off_mm"] = np.asarray(tw["root_off_mm"], dtype=np.float32)
        if tw.get("cam_err_mm"):
            twin["twin_cam_err_mm"] = np.asarray(tw["cam_err_mm"], dtype=np.float32)
            twin["twin_cam_err_deg"] = np.asarray(tw["cam_err_deg"], dtype=np.float32)
    if err is not None:  # <OUTROOT>/<block>/<cell> -> <OUTROOT>_errors/<block>/<cell> (readout never sees it)
        dd = os.path.normpath(d)
        d = os.path.join(os.path.dirname(os.path.dirname(dd)) + "_errors", *dd.split(os.sep)[-2:])
        os.makedirs(d, exist_ok=True)
    final = os.path.join(d, f"t{task_id}_e{ep}_i{init_id}.npz")
    tmp = os.path.join(d, f".tmp_{os.getpid()}_t{task_id}_e{ep}_i{init_id}.npz")  # never matches t*_e*_i*.npz
    np.savez_compressed(
        tmp,
        success=success, task=task_id, ep=ep, init=init_id,
        action=np.asarray(trace["action"], dtype=np.float32),
        eef_pos=np.asarray(trace["eef_pos"], dtype=np.float32),
        eef_quat=np.asarray(trace["eef_quat"], dtype=np.float32),
        gripper_qpos=np.asarray(trace["gripper_qpos"], dtype=np.float32),
        qpos=np.asarray(trace["qpos"], dtype=np.float64),
        joint_names=np.asarray(trace["joint_names"]),
        ref_body_names=np.asarray(trace["ref_body_names"]),
        ref_body_pos=trace["ref_body_pos"], ref_body_quat=trace["ref_body_quat"],
        **contacts,
        host=host, gpu=gpu, align_json=json.dumps(getattr(env, "align_info", None)),
        **pg, **twin,
        query_step=np.asarray([q["step"] for q in pg_q], np.int32),
        query_chunk=np.asarray([q["chunk"] for q in pg_q], np.float32),
        mi_noise_md5=np.asarray([q["noise_md5"] for q in pg_q]),
        noise_seed=np.asarray([q["noise_seed"] for q in pg_q], np.int64),
        mi_noise_master=np.int64(int(os.environ["MI_NOISE_MASTER"])), mi_model_tag=os.environ["MI_MODEL_TAG"],
        pg_model=args.model, pg_suite=args.suite, pg_replan=np.int32(args.replan),
        pg_chunk_len=np.int32(chunk_len if chunk_len is not None else -1),
        pg_server_cfg=json.dumps(srv_cfg), pg_error=json.dumps(err),
        server_host=os.environ.get("NUS_SRV_HOST", args.host), server_gpu=os.environ.get("NUS_SRV_GPU", ""),
        embody_dir=os.environ.get("EMBODY_DIR", ""), pg_blind=True,
        pg_gl_renderer=json.dumps(getattr(env, "_mi_glr", None)),
        pg_cell_env=json.dumps({k: os.environ.get(k) for k in ("BODY", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY",
                                                               "TWIN_MOUNT", "PG_MODE", "PG_WINDOW", "PG_LIFT_M")}),
    )
    os.replace(tmp, final)


if __name__ == "__main__":
    main()
