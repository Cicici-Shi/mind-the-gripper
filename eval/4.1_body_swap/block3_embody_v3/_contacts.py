"""Replay every episode of one condition (its own recorded actions, same init + fixtures) and log
arm-vs-environment contacts and hand stalls. Usage: python -m _contacts COND BODY OUT.jsonl [ENV=VAL...]"""
import glob, json, os, sys
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite

cond, body, out = sys.argv[1:4]
for kv in sys.argv[4:]:
    k, v = kv.split("="); os.environ[k] = v
suite = OriginalSpatialSuite()
files = sorted(glob.glob(os.path.expanduser(f"$PROJ_ROOT/runs/{cond}/t*_e*_i*.npz")))
done_files = set()
if os.path.exists(out):
    done_files = {json.loads(l)["file"] for l in open(out)}
envs = {}
def inner(e):
    for _ in range(6):
        if hasattr(type(e), "_get_observations"):
            return e
        e = e.env
for f in files:
    b = os.path.basename(f)
    if b in done_files:
        continue
    z = np.load(f, allow_pickle=True)
    task = int(z["task"])
    t = suite.get_task(task)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64); ref.seed(0)
    env = BodyEnv(ref, bddl, body, 64, 10, log=lambda m: None, render=False)
    env.reset()
    ref.sim.model.body_pos[:] = z["ref_body_pos"]; ref.sim.model.body_quat[:] = z["ref_body_quat"]; ref.sim.forward()
    env.set_init_state(suite.get_task_init_states(task)[int(z["init"])])
    e = inner(env.env); sim = e.sim; m = sim.model; d = sim.data
    grip_prefix = e.robots[0].gripper.naming_prefix
    pairs = {}; first = None; steps_contact = 0; stall = 0; done_any = False
    pos = []
    for k, a in enumerate(z["action"]):
        _, _, done, _ = env.step(a.tolist()); done_any = done_any or bool(done)
        hit = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""; b2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
            for x, y in ((b1, b2), (b2, b1)):
                if x.startswith("robot0") and not y.startswith(("robot0", grip_prefix, "mount0")):
                    hit = True; pairs[f"{x}|{y}"] = pairs.get(f"{x}|{y}", 0) + 1
        if hit:
            steps_contact += 1
            if first is None: first = k
        pos.append(d.site_xpos[e.robots[0].eef_site_id].copy())
        if k >= 10 and np.linalg.norm(z["action"][k-10:k, :3], axis=1).mean() > 0.5 and np.linalg.norm(pos[-1] - pos[-11]) < 0.005:
            stall += 1
    rec = {"file": b, "task": task, "ep": int(z["ep"]), "success": bool(z["success"]), "success_replay": done_any,
           "steps": len(z["action"]), "arm_env_contact_steps": steps_contact, "first_contact": first,
           "stall_steps": stall, "pairs": pairs}
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    env.close()
print("done", cond)
