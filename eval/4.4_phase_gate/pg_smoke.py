"""phasegate smoke (no policy, no outcome): scripted descend / close / lift on one init state of libero_spatial
task 0, for both modes and all six windows (post added for phasegate-confirm). Interventions are render-only, so the physics of every run must be
identical; the "none" and "all" runs of a mode give the two reference image sequences. Checks per (mode, window):
  - the three phases are reached in order and the gate equals (window == phase) at every observation
    (post: open in grasp and transport, i.e. phase >= 1);
  - qpos at every step equals the "none" run (render-only);
  - gate closed -> both policy images equal the "none" run's frame; gate open -> they equal the "all" run's frame
    (tolerance NOISE_PX differing pixels per view: same-state renders differ by isolated pixels on this cluster);
  - "all" differs from "none" in at least one view on >= 90% of the observations;
  - gripper alpha values are back to their originals after every step.
Saves one strip per (mode, window): agentview | wrist at the first observation of each phase.
Usage (GPU node, EGL): python tools/pg_smoke.py OUT_DIR
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

IMG = ("agentview_image", "robot0_eye_in_hand_image")
SCRIPT = [[0, 0, -0.5, 0, 0, 0, -1]] * 20 + [[0, 0, 0, 0, 0, 0, 1]] * 12 + [[0, 0, 0.8, 0, 0, 0, 1]] * 25
ENVS = {"look": {"TWIN_RENDER": "1", "TWIN_BODY": "PandaR85", "TWIN_MOUNT": "sunk"}, "erase": {}}
NOISE_PX = 20


def rollout(mode, window):
    from embody.body_env import BodyEnv
    from embody.original import OriginalSpatialSuite
    for k in ("TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT"):
        os.environ.pop(k, None)
    os.environ.update(ENVS[mode], PG_MODE=mode, PG_WINDOW=window, COPY_GRIP_CMD="1")
    suite = OriginalSpatialSuite()
    t = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    env = BodyEnv(ref, bddl, "Panda", 256, 10, log=lambda m: print(m, flush=True))
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    m = env.env.sim.model
    palm, fingers = env._grip_parts(m)
    alpha0 = [float(m.geom_rgba[g][3]) for g in palm + fingers]
    frames, qpos, alpha_ok = [], [], True
    for k in range(len(SCRIPT) + 1):
        frames.append({key: np.array(obs[key], copy=True) for key in IMG})
        qpos.append(env.env.sim.data.qpos.copy())
        alpha_ok &= [float(m.geom_rgba[g][3]) for g in palm + fingers] == alpha0
        if k < len(SCRIPT):
            obs, _, _, _ = env.step(SCRIPT[k])
    out = {"frames": frames, "qpos": np.asarray(qpos), "gates": list(env.pg.gates), "phases": list(env.pg.phases),
           "t_close": env.pg.t_close, "t_lift": env.pg.t_lift, "alpha_ok": alpha_ok, "fields": env.pg_fields()}
    env.close()
    return out


def ndiff(a, b):
    return [int(np.any(a[k] != b[k], axis=2).sum()) for k in IMG]


def main(out_dir):
    from embody import phasegate
    os.makedirs(out_dir, exist_ok=True)
    all_ok = True
    for mode in phasegate.MODES:
        runs = {w: rollout(mode, w) for w in phasegate.WINDOWS}
        none, full = runs["none"], runs["all"]
        sep = [sum(ndiff(a, b)) > NOISE_PX for a, b in zip(none["frames"], full["frames"])]
        for w, r in runs.items():
            bad = []
            if sorted(set(r["phases"])) != [0, 1, 2]:
                bad.append(f"phases reached {sorted(set(r['phases']))}")
            if not r["alpha_ok"]:
                bad.append("gripper alpha not restored")
            if not np.array_equal(r["qpos"], none["qpos"]):
                bad.append(f"physics differs from the none run (max {np.abs(r['qpos'] - none['qpos']).max():.2e})")
            if r["phases"] != none["phases"]:
                bad.append("phase sequence differs from the none run")
            worst = [0, 0]
            for k, (g, ph) in enumerate(zip(r["gates"], r["phases"])):
                if g != (w == "all" or w == phasegate.PHASES[ph] or (w == "post" and ph >= 1)):
                    bad.append(f"obs {k}: gate {g} for phase {ph}")
                d = ndiff(r["frames"][k], (full if g else none)["frames"][k])
                worst = [max(a, b) for a, b in zip(worst, d)]
                if max(d) > NOISE_PX:
                    bad.append(f"obs {k}: gate {'open' if g else 'closed'} but {d} px differ from the {'all' if g else 'none'} run")
            if w == "all" and np.mean(sep) < 0.9:
                bad.append(f"all vs none differ on only {np.mean(sep):.2f} of the observations")
            if w == "none" and mode == "erase":  # the direct render used for the first erased frame has the sensor's orientation
                d0 = ndiff(r["frames"][0], full["frames"][0])
                same_rows = [int(np.any(r["frames"][0][key] != full["frames"][0][key], axis=2).any(axis=1).sum()) for key in IMG]
                print(f"PG_SMOKE erase t0: none vs all differ in {d0} px over {same_rows} rows (a flipped render would differ almost everywhere)")
                if max(d0) > 0.5 * 256 * 256:
                    bad.append(f"first erased frame differs from the plain frame in {d0} px: orientation?")
            f = r["fields"]
            if len(f["pg_gate"]) != len(SCRIPT) + 1 or len(f["pg_phase"]) != len(SCRIPT) + 1:
                bad.append("dump field lengths")
            first = [r["phases"].index(p) for p in sorted(set(r["phases"]))]
            Image.fromarray(np.concatenate([np.concatenate([r["frames"][k][key][::-1, ::-1] for key in IMG], axis=1)
                                            for k in first], axis=0)).save(os.path.join(out_dir, f"{mode}_{w}.png"))
            print(f"PG_SMOKE {mode} {w}: t_close {r['t_close']} t_lift {r['t_lift']} gated {int(sum(r['gates']))} "
                  f"max_px_vs_reference {worst} all_vs_none_sep {np.mean(sep):.2f} "
                  f"{'PASS' if not bad else 'FAIL ' + ' | '.join(bad[:4])}", flush=True)
            all_ok &= not bad
    print("PG_SMOKE_ALL", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
