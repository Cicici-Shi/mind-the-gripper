"""Profile TWIN_CAM_SYNC=1 paint cost (models1 throughput gate), 2026-11-04. No policy: zero actions, task 0 ep 0."""
import cProfile, os, pstats, sys, time, copy
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ["EMBODY_DIR"]))
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite
from embody import snapshot as _snap
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = OriginalSpatialSuite(); task = suite.get_task(0); inits = suite.get_task_init_states(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
np.random.seed(7)
ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); ref.seed(0)
env = BodyEnv(ref, bddl, os.environ["BODY"], 256, 10, log=lambda m: None)
env.reset(); obs = env.set_init_state(inits[42])
d = env.env.sim.data
t = time.time(); [copy.copy(d._data) for _ in range(5)]; print("copy.copy(MjData) s/call", (time.time() - t) / 5)
k = copy.copy(d._data)
t = time.time(); [_snap._copy_mjdata(k, d._data) for _ in range(5)]; print("_copy_mjdata s/call", (time.time() - t) / 5)
print("nstack/narena", getattr(env.env.sim.model._model, "nstack", None), getattr(env.env.sim.model._model, "narena", None))
pr = cProfile.Profile(); pr.enable(); t = time.time()
for i in range(20):
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
print("SYNC", os.environ.get("TWIN_CAM_SYNC"), "s/step", (time.time() - t) / 20)
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
