"""models1 (2026-11-04): faster TWIN_CAM_SYNC=1 snapshot for embody_toy BodyEnv, applied by monkeypatch (embody_toy is
read-only). Original (embody_toy/embody/body_env.py BodyEnv._twin_paint): keep = copy.copy(MjData) (mj_copyData of a
166 MB arena, 1.26 s per paint on node09 T4, ~8x episode wall time), forward, paint, then snapshot._copy_mjdata(d, keep).
_copy_mjdata copies every public ndarray field of src into dst where dst's field is a writeable ndarray of the same
shape, plus time. Here the snapshot keeps exactly those src arrays as numpy copies (shapes as of the snapshot, as in
copy.copy) and the restore applies the same rule, so the restored MjData is bitwise the same as with the original.
Verified bitwise by tools/fastsync_check.py (WORKLOG_vlazoo 8.x). Import AFTER EMBODY_DIR is on sys.path; call patch().
"""
import os
import numpy as np


def _snapshot(d):
    keep = {}
    for name in dir(d):
        if name.startswith("_"):
            continue
        try:
            v = getattr(d, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            keep[name] = v.copy()
    return keep, d.time


def _restore(dst, snap):
    keep, t = snap
    for name, v in keep.items():
        try:
            d = getattr(dst, name)
            if isinstance(d, np.ndarray) and d.shape == v.shape and d.flags.writeable:
                d[...] = v
        except Exception:
            pass
    dst.time = t


def patch():
    from embody.body_env import BodyEnv
    if getattr(BodyEnv, "_m1_fastsync", False):
        return
    def _twin_paint(self, obs):
        if os.environ.get("TWIN_CAM_SYNC") != "1":
            return self._twin_paint_inner(obs)
        d = self.env.sim.data
        snap = _snapshot(d._data)
        self.env.sim.forward()
        try:
            return self._twin_paint_inner(obs)
        finally:
            _restore(d._data, snap)
    BodyEnv._twin_paint_orig = BodyEnv._twin_paint
    BodyEnv._twin_paint = _twin_paint
    BodyEnv._m1_fastsync = True
