"""OVLA GPU re-run with per-step physical qpos (Fable add1 2026-11-05, section 4 item 4; ruling (c)).

Re-runs OpenVLA on given (task, episode) keys of one models1 cell through the FROZEN common/zoo_client.py (imported,
never modified) with exactly the cell switches of `tools/zoo_cells.sh vars <cell>`, EMBODY_DIR=embody_toy,
M1_FASTSYNC=1, the same init selection (zoo_client: RandomState(5478 + task).permutation) and the same global
np.random stream (the client is run with --tasks all --trials 5 --ep-offset <shard> --skip-done, every key that is not
re-run is pre-registered as a skip stub, so fixture placement draws are those of the original shard run).
Extra, recorded at run time by wrapping BodyEnv.step / set_init_state (or the SRC env), in the layout of the OFTi dumps
(so tools/reach_diag_m1.py can read them):
  qpos[i], eef_pos[i], eef_quat[i], gripper_qpos[i], action[i]   state BEFORE policy action i, the action itself
  twin_pos_err_mm[j], twin_rot_err_deg[j], twin_cam_err_mm/deg[j]  BodyEnv.twin_stats per paint (j=0 initial frame,
                                                                    j=k after step k-1; so index i matches qpos[i])
  twin_q[j]                                                        BodyEnv's own twin IK solution at paint j
  qacc_warn[i]                                                     cumulative MuJoCo mjWARN_BADQACC count after step i
  qpos_last                                                        physical qpos after the last step
Output: out/ovla_rerun/<tag>/<cell>/t{task}_e{ep}_i{init}.npz and out/ovla_rerun/<tag>/<cell>_e{off}_p{part}.jsonl.
NEVER writes runs/openvla/ (SKIP_DONE there would take re-runs for collected keys). Outcome firewall: the `success`
key is removed from the client record before it is written; no npz has a success member; nothing prints outcomes.

Subcommands
  keys      freeze the GTw (g_tw_fc) over-1-mm keys of out/ik_diag_m1_openvla_libero_spatial.json -> <outdir>/keys.json
  run       one client process: --cell C --offset O --keys t:e,t:e --part P --host H --port P
  selftest  no server, no policy: build the cell's env, 5 DUMMY steps, check the recorded fields (field check only)
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
TOY = os.path.expanduser("$PROJ_ROOT/embody_toy")
TRIALS = 5


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cell_env(cell):
    """BODY and VAR=VAL list exactly as tools/zoo_cells.sh (the frozen single source)."""
    v = subprocess.run(["bash", os.path.join(Z, "tools", "zoo_cells.sh"), "vars", cell], capture_output=True,
                       text=True, check=True).stdout.split()
    return v[0], dict(x.split("=", 1) for x in v[1:])


# ------------------------------------------------------------------ keys
def cmd_keys(a):
    src = os.path.join(Z, "out", "ik_diag_m1_openvla_libero_spatial.json")
    d = json.load(open(src))
    c = d["cells"]["g_tw_fc"]
    assert c["complete_shards"] == [0], c["complete_shards"]
    # same loader rule as ik_diag_m1.load_jsonl: first error-free record per key; success popped at once
    recs = {}
    for f in sorted(glob.glob(os.path.join(Z, "runs", "openvla", "g_tw_fc_e*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.pop("success", None)
            if r.get("error") is not None:
                continue
            k = (int(r["task"]), int(r["episode"]))
            if k[1] >= TRIALS or k in recs:  # shard 0 only (the shard ik_diag used)
                continue
            recs[k] = np.asarray((r.get("twin") or {}).get("pos_err_mm_q") or [], float)
    over = sorted(k for k, pe in recs.items() if (pe > 1).any())
    # consistency with the json: count, the printed top 10 and the per-task table
    assert len(recs) == c["keys_used"] == 50, len(recs)
    assert len(over) == c["eps_with_gt1mm"], (len(over), c["eps_with_gt1mm"])
    top = {tuple(e["key"]) for e in c["top_episodes"]}
    assert top <= set(over), top - set(over)
    for t, v in c["by_task"].items():
        assert v["eps_with_over"] == sum(1 for k in over if k[0] == int(t)), t
        assert v["over_frames"] == int(sum((recs[k] > 1).sum() for k in over if k[0] == int(t))), t
    for e in c["top_episodes"]:
        assert e["over1"] == int((recs[tuple(e["key"])] > 1).sum()), e["key"]
    out = {"cell": "g_tw_fc", "suite": "libero_spatial", "offset": 0, "keys": [list(k) for k in over],
           "rule": "first error-free record per key in runs/openvla/g_tw_fc_e*.jsonl, shard 0 (ep 0-4), any query-frame "
                   "twin pos_err_mm_q > 1 mm (= ik_diag_m1 eps_with_gt1mm); cross-checked against ik_diag json "
                   "eps_with_gt1mm, top_episodes (key, over1) and by_task",
           "source": {"ik_diag_json": src, "ik_diag_json_sha256": sha(src),
                      "jsonl_sha256": {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(Z, "runs", "openvla", "g_tw_fc_e0.jsonl")))}},
           "smoke_src_key": [2, 1]}
    os.makedirs(a.outdir, exist_ok=True)
    p = os.path.join(a.outdir, "keys.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"{len(over)} keys {over} -> {p}")


# ------------------------------------------------------------------ recorder
class Rec:
    """Per-episode buffer; one active episode per process (zoo_client runs episodes sequentially)."""
    buf = None
    benv = None  # BodyEnv instance of the active episode (None for SRC)
    n_wait_left = 0
    last_obs = None


def _qacc(sim):
    try:
        import mujoco
        return int(sim.data._data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
    except Exception:
        return -1


def _start(obs, phys_env, benv, n_wait):
    m = phys_env.sim.model
    Rec.buf = {"action": [], "eef_pos": [], "eef_quat": [], "gripper_qpos": [], "qpos": [], "qacc_warn": [],
               "twin_q": [], "joint_names": [m.joint_id2name(j) or "" for j in range(m.njnt)],
               "qacc_warn0": _qacc(phys_env.sim)}
    Rec.benv, Rec.n_wait_left, Rec.last_obs, Rec.phys = benv, n_wait, obs, phys_env
    if benv is not None and getattr(benv, "twin", None) is not None:
        Rec.buf["twin_q"].append(np.array(benv._twin_q, dtype=np.float64, copy=True))


def _pre(action):
    b, o = Rec.buf, Rec.last_obs
    if b is None or Rec.n_wait_left > 0:
        return
    b["action"].append(np.asarray(action, np.float32))
    b["eef_pos"].append(np.array(o["robot0_eef_pos"], np.float32))
    b["eef_quat"].append(np.array(o["robot0_eef_quat"], np.float32))
    b["gripper_qpos"].append(np.array(o["robot0_gripper_qpos"], np.float32))
    b["qpos"].append(Rec.phys.sim.data.qpos.copy())


def _post(obs):
    b = Rec.buf
    if b is None:
        return
    Rec.last_obs = obs
    if Rec.n_wait_left > 0:  # SRC: the client's own wait steps are not policy steps (OFT dump layout)
        Rec.n_wait_left -= 1
        return
    b["qacc_warn"].append(_qacc(Rec.phys.sim))
    if Rec.benv is not None and getattr(Rec.benv, "twin", None) is not None:
        b["twin_q"].append(np.array(Rec.benv._twin_q, dtype=np.float64, copy=True))


def _flush(rec, cell_dir):
    b = Rec.buf
    if b is None:
        return None
    t, e, i = int(rec["task"]), int(rec["episode"]), int(rec["init"])
    out = dict(task=t, ep=e, init=i, action=np.asarray(b["action"], np.float32).reshape(-1, 7),
               eef_pos=np.asarray(b["eef_pos"], np.float32).reshape(-1, 3),
               eef_quat=np.asarray(b["eef_quat"], np.float32).reshape(-1, 4),
               gripper_qpos=np.asarray(b["gripper_qpos"], np.float32),
               qpos=np.asarray(b["qpos"], np.float64), joint_names=np.asarray(b["joint_names"]),
               qpos_last=Rec.phys.sim.data.qpos.copy(), qacc_warn=np.asarray(b["qacc_warn"], np.int32),
               qacc_warn0=np.int32(b["qacc_warn0"]))
    tw = getattr(Rec.benv, "twin_stats", None) if Rec.benv is not None else None
    if tw is not None:
        out.update(twin_pos_err_mm=np.asarray(tw["pos_err_mm"], np.float32),
                   twin_rot_err_deg=np.asarray(tw["rot_err_deg"], np.float32),
                   twin_q=np.asarray(b["twin_q"], np.float64), twin_stale=np.int32(tw["stale"]),
                   twin_retries=np.int32(tw.get("retries", 0)))
        if tw.get("cam_err_mm"):
            out.update(twin_cam_err_mm=np.asarray(tw["cam_err_mm"], np.float32),
                       twin_cam_err_deg=np.asarray(tw["cam_err_deg"], np.float32))
    os.makedirs(cell_dir, exist_ok=True)
    p = os.path.join(cell_dir, f"t{t}_e{e}_i{i}.npz")
    np.savez_compressed(p, **out)
    Rec.buf = None
    return p


def install(zc, body, cell_dir):
    """Runtime wrappers only; zoo_client.py and embody_toy files are not touched."""
    if body:
        from embody.body_env import BodyEnv
        o_init, o_step = BodyEnv.set_init_state, BodyEnv.step

        def set_init_state(self, init_state):
            obs = o_init(self, init_state)
            _start(obs, self.env, self, 0)
            return obs

        def step(self, action):
            _pre(action)
            r = o_step(self, action)
            _post(r[0])
            return r
        BodyEnv.set_init_state, BodyEnv.step = set_init_state, step
    else:
        base = zc.OffScreenRenderEnv

        class RecEnv(base):
            def set_init_state(self, init_state):
                obs = super().set_init_state(init_state)
                _start(obs, self, None, zc.NUM_STEPS_WAIT)
                return obs

            def step(self, action):
                _pre(action)
                r = super().step(action)
                _post(r[0])
                return r
        zc.OffScreenRenderEnv = RecEnv

    class JsonShim:  # zoo_client writes each record with json.dumps(rec): dump the npz there, drop the outcome
        loads = staticmethod(json.loads)
        JSONDecodeError = json.JSONDecodeError

        @staticmethod
        def dumps(obj, *a, **k):
            if isinstance(obj, dict) and "task" in obj and "episode" in obj and "init" in obj:
                obj = dict(obj)
                obj.pop("success", None)
                p = _flush(obj, cell_dir)
                obj["rerun_npz"] = os.path.relpath(p, Z) if p else None
            return json.dumps(obj, *a, **k)
    zc.json = JsonShim


def import_client(cell):
    body, env = cell_env(cell)
    os.environ.update(env)
    os.environ["EMBODY_DIR"] = TOY
    os.environ["M1_FASTSYNC"] = "1"
    os.environ.setdefault("HIDE_ARM", "0")
    os.environ.setdefault("COPY_GRIP_CMD", "1")
    f = os.path.join(Z, "common", "zoo_client.py")
    spec = importlib.util.spec_from_file_location("zoo_client", f)
    zc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zc)
    return zc, ("" if body == "SRC" else body), env, sha(f)


# ------------------------------------------------------------------ run
def cmd_run(a):
    zc, body, env, zsha = import_client(a.cell)
    keys = {tuple(int(x) for x in k.split(":")) for k in a.keys.split(",") if k}
    assert all(a.offset <= e < a.offset + TRIALS for _, e in keys), keys
    cell_dir = os.path.join(a.outdir, a.cell)
    os.makedirs(cell_dir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.cell}_e{a.offset}_p{a.part}.jsonl")
    done = set()
    if os.path.exists(out):  # resumable: keep finished keys, stubs are rewritten below anyway
        for line in open(out):
            r = json.loads(line)
            if not r.get("stub") and r.get("error") is None:
                done.add((r["task"], r["episode"]))
    with open(out, "a") as f:
        for t in range(10):
            for e in range(a.offset, a.offset + TRIALS):
                if (t, e) not in keys and (t, e) not in done:
                    f.write(json.dumps({"task": t, "episode": e, "error": None, "stub": "skip (not re-run)"}) + "\n")
    print(f"RERUN cell={a.cell} body={body or 'SRC'} env={env} offset={a.offset} keys={sorted(keys)} "
          f"zoo_client_sha256={zsha} out={out}", flush=True)
    install(zc, body, cell_dir)
    sys.argv = ["zoo_client.py", "--proto", "raw", "--model", "openvla", "--suite", "libero_spatial",
                "--host", a.host, "--port", str(a.port), "--trials", str(TRIALS), "--ep-offset", str(a.offset),
                "--skip-done", "--out", out] + (["--body", body] if body else [])
    zc.main()


# ------------------------------------------------------------------ selftest (field check, no policy)
def cmd_selftest(a):
    zc, body, env, zsha = import_client(a.cell)
    cell_dir = os.path.join(a.outdir, "selftest_" + a.cell)
    install(zc, body, cell_dir)
    suite = zc.OriginalSpatialSuite()
    task = suite.get_task(0)
    bddl = os.path.join(zc.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    np.random.seed(7)
    e = zc.OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=zc.ENV_RES, camera_widths=zc.ENV_RES)
    e.seed(0)
    if body:
        from embody.body_env import BodyEnv
        e = BodyEnv(e, bddl, body, zc.ENV_RES, zc.NUM_STEPS_WAIT, log=lambda m: print(m, flush=True))
    e.reset()
    e.set_init_state(suite.get_task_init_states(0)[0])
    t = zc.NUM_STEPS_WAIT if getattr(e, "skip_wait", False) else 0
    n = 0
    while n < a.steps:
        e.step(zc.DUMMY)  # zero motion, gripper open; not a policy step of any protocol cell
        n += t >= zc.NUM_STEPS_WAIT
        t += 1
    p = _flush({"task": 0, "episode": -1, "init": 0}, cell_dir)
    z = np.load(p)
    print("SELFTEST", a.cell, "fields", {k: z[k].shape for k in z.files if k != "joint_names"}, flush=True)
    nq = z["qpos"].shape
    assert nq[0] == a.steps and z["action"].shape == (a.steps, 7) and z["eef_pos"].shape == (a.steps, 3)
    assert z["qacc_warn"].shape == (a.steps,)
    arm = [i for i, s in enumerate(z["joint_names"]) if str(s).startswith("robot0_joint")]
    assert len(arm) == 7, arm
    if body and "TWIN_RENDER" in env:
        assert z["twin_pos_err_mm"].shape == (a.steps + 1,) and z["twin_rot_err_deg"].shape == (a.steps + 1,)
        assert z["twin_q"].shape == (a.steps + 1, 7)
        print("SELFTEST twin pos mm max", float(z["twin_pos_err_mm"].max()), "rot deg max", float(z["twin_rot_err_deg"].max()))
    print("SELFTEST OK", p, "zoo_client_sha256", zsha, flush=True)
    e.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["keys", "run", "selftest"])
    ap.add_argument("--outdir", default=os.path.join(Z, "out", "ovla_rerun"))
    ap.add_argument("--cell", default="g_tw_fc")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--keys", default="")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    a.outdir = os.path.abspath(a.outdir)
    assert "/runs/" not in a.outdir + "/" and os.path.realpath(a.outdir).startswith(os.path.realpath(os.path.join(Z, "out"))), a.outdir
    {"keys": cmd_keys, "run": cmd_run, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
