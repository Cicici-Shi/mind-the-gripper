import os, sys, json, numpy as np, tempfile
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_v3"))
from embody import contacts, robots  # noqa
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite, OriginalObjectSuite
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
for Suite in (OriginalSpatialSuite, OriginalObjectSuite):
    su = Suite(); t = su.get_task(3 if Suite is OriginalSpatialSuite else 0)
    print(Suite.__name__, su.n_tasks, [x.name[:40] for x in su.tasks][:3])
    b = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    for body, env_extra in (("SRC", {}), ("UR5e", {"NO_FOREARM_CONTACT": "1"}), ("Kinova3", {}), ("PandaR85", {}), ("Panda", {})):
        os.environ.update(env_extra)
        ref = ControlEnv(bddl_file_name=b, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False); ref.seed(0)
        env = ref if body == "SRC" else BodyEnv(ref, b, body, 64, 10, log=lambda m: None, render=False)
        env.reset(); env.set_init_state(su.get_task_init_states(0)[0])
        if body == "SRC":
            for _ in range(10): env.step([0]*6+[-1])
        log = contacts.make(env)
        for k in range(30):
            a = np.array([0.3, 0, -0.5, 0, 0, 0, -1.0]); env.step(a.tolist()); log.update(a)
        f = log.npz_fields()
        np.savez_compressed(os.path.join(tempfile.gettempdir(), "x.npz"), **f)
        print(body, env_extra, json.dumps(log.summary())[:300])
        for k in env_extra: os.environ.pop(k)
        if body != "SRC":
            from embody import align
            f, fr = align.frames(env.env), align.frames(env.ref)
            print("   ", body, "base diff mm %.3f" % (np.linalg.norm(f["base_pos"] - fr["base_pos"]) * 1000), "align", {k: round(v, 4) if isinstance(v, float) else v for k, v in env.align_info.items()})
