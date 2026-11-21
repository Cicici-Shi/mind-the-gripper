"""maskinfo-pi (2026-11-08): the maskinfo cells on the openpi pi0 / pi0.5 clients (protocol_maskinfo_pi.md, Astra
rulings 2026-11-08). embody/maskinfo.py (3d301bad...) and embody/maskrepl.py (dffe5b04...) are byte copies of the
frozen maskinfo files and are NOT modified; everything below is a subclass plus a model-input swap.

Differences from the frozen MaskInfoHook (all configurable, no value is hard coded in the hook):
  * query cadence: query_every (pi0 / pi0.5: 5; frozen OFT: 8). The frozen constructor asserts lag % 8 == 0 and
    _lag_G looks the lag frame up BY INDEX (q - lag // 8), which in a 5-step loop silently becomes the wrong step
    (feasibility R1). Here the lag frame is looked up BY STEP NUMBER in Ghist, and it is ASSERTED that the frame
    actually taken is step (t - lag) (or frame 0 while t < lag: hold), where t is the step the caller announced
    with arm(step). lag % query_every == 0 is asserted (no per-step G caching; protocol 2.3 / ruling 3).
    finish() asserts step == the announced step and step % query_every == 0, so iou_lag / open_changed_lag
    (manipulation check item 7, found by step in the frozen finish) always has its history frame for t >= lag.
  * model input path (MI_PREP=openpi): the frozen hook's self checks and the section 5 224 quantities use OFT's
    path (rotate 180, JPEG + lanczos 224, centre crop 0.9). With openpi the policy sees rotate 180 ->
    openpi_client.image_tools.resize_with_pad(224, 224) (256 -> 224 square: bilinear, no padding) -> uint8, no
    crop. install_prep("openpi") swaps to_model_input / mask_to_model in BOTH embody.maskrepl and embody.maskinfo
    (the frozen modules import them by name). note_model_input reads the openpi request keys.
  * C8 = seg-BG ("Trace-style BG", ruling 6): the frozen maskrepl R1 path (region seg = G, fill background_fill =
    ring mean, ring width 8 = TRACE["bg_ring_width"]), both views; the mask is the same G as C1.
  * proprio check (ruling 11): at every armed _update_observables call the state observables (eef pos / quat,
    gripper qpos) are copied right BEFORE and right AFTER the frozen edit; finish() compares, for the same
    observation moment, pre == post == the returned observation (after BodyEnv._fix's quaternion sign). The client
    compares the 8-d state it actually sends with the one built from the pre-edit copy (row["mi_S"]).
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

from embody import maskinfo as mi
from embody import maskrepl as mr
from embody.maskinfo import MaskInfoHook

MASKINFO_SHA = "3d301badbb870c9a5491fa18a8d74136f13da2bcf4f5ff29378379b660e14653"
MASKREPL_SHA = "dffe5b0420685a544c84510fbd7f5755c0ce694bf88358ba2249f4ca5c93b257"
PI_CELLS = dict(mi.MI_CELLS)
PI_CELLS["C8"] = {"region": "seg", "fill": "bg", "overlay": "none", "protect": "none", "bgmask": "none",
                  "maskrepl": "R1 (Trace-style BG: ring-mean fill, ring 8, mask = G as C1)"}
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- openpi model-input path
def to_model_input_openpi(img):
    """openpi LIBERO client path (zoo_client make_element): rotate 180, resize_with_pad(224, 224), uint8.
    Returns (c, r) like the frozen OFT function; both are the 224 image the server receives (no crop)."""
    from openpi_client import image_tools
    r = np.ascontiguousarray(np.asarray(img)[::-1, ::-1])
    r = image_tools.convert_to_uint8(image_tools.resize_with_pad(r, 224, 224))
    r = np.asarray(r, dtype=np.uint8)
    return r, r


def mask_to_model_openpi(mask, size=224):
    """Geometric twin for masks: rotate 180, bilinear resize to 224 (256 -> 224 square, no pad, no crop), >= 128."""
    m = Image.fromarray(np.where(np.asarray(mask, bool)[::-1, ::-1], 255, 0).astype(np.uint8), mode="L")
    return np.asarray(m.resize((size, size), Image.BILINEAR)) >= 128


PREP = {"openpi": (to_model_input_openpi, mask_to_model_openpi), "oft": (mr.to_model_input, mr.mask_to_model)}
_ACTIVE = {"prep": None}


def install_prep(name):
    """Swap the model-input functions used by the frozen hooks (module globals; files untouched)."""
    f, g = PREP[name]
    for mod in (mr, mi):
        mod.to_model_input, mod.mask_to_model = f, g
    _ACTIVE["prep"] = name


# ---------------------------------------------------------------- hook
class MaskInfoPiHook(MaskInfoHook):
    def __init__(self, benv, cell, res=256, metrics_path=None, log=print, lag_steps=15, outline_width=3,
                 protect_set="interactive", query_every=5, prep="openpi", protocol_path=None):
        query_every, lag_steps = int(query_every), int(lag_steps)
        assert query_every > 0 and lag_steps > 0 and lag_steps % query_every == 0, (lag_steps, query_every)
        assert protect_set in mi.PROTECT_SETS, protect_set
        install_prep(prep)
        self.query_every, self.prep = query_every, prep
        self.protocol_path = protocol_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocol_maskinfo_pi.md")
        self.lag_steps, self.outline_width, self.protect_set = lag_steps, int(outline_width), protect_set
        self.det_check = False
        self._next_step = None
        mr.MaskHook.__init__(self, benv, cell, res=res, metrics_path=metrics_path, log=log)  # skips the 8-step assert

    def set_cell(self, cell):
        self.cell = cell
        self.mi = None
        if cell is None:
            self.region, self.fill, self.views, self.spec = "none", "none", (), None
            return
        c = PI_CELLS[cell]
        self.mi = c
        self.region = c["region"]
        self.fill = {"alpha0": "none"}.get(c["fill"], c["fill"])
        self.views = mr.VIEWS["both"]
        s = mi.mi_spec("C0", self.lag_steps, self.outline_width, self.protect_set)  # shared fields of the frozen spec
        s.update(c, cell=cell, query_every=self.query_every, lag_steps=self.lag_steps,
                 lag_lookup="by step number in Ghist, asserted == t - lag (t < lag: frame 0)",
                 model_prep=self.prep, protocol="protocol_maskinfo_pi.md", protocol_sha256=_sha(self.protocol_path),
                 maskinfo_sha256=MASKINFO_SHA, maskrepl_sha256=MASKREPL_SHA,
                 maskinfo_pi_sha256=_sha(os.path.abspath(__file__)))
        if cell == "C8":
            s["trace_fill"] = mr.TRACE_MODE["bg"]
        self.spec = s

    def begin_episode(self, task, ep):
        super().begin_episode(task, ep)
        if self.spec is not None:
            self.spec = dict(self.spec, cell=self.cell)
        self._next_step = None

    def arm(self, step=None):
        """step = the action count after the coming env.step (0 for set_init_state): the step whose observation
        feeds the next query. Required when a cell is set."""
        if self.cell is not None:
            assert step is not None, "MaskInfoPiHook.arm(step) needs the step number"
            step = int(step)
            assert step % self.query_every == 0, f"armed at step {step}, not a multiple of {self.query_every}"
            self._next_step = step
        super().arm()

    def _snap_state(self):
        obs = self.rs._observables
        return {k: np.array(obs[k].obs, copy=True) for k in STATE_KEYS if k in obs}

    def _post_call(self):
        pre = self._snap_state()
        super()._post_call()
        post = self._snap_state()
        if self.cap is not None:
            self.cap["_S"] = (pre, post)

    def _lag_G(self, cam, c):
        if self.mi["overlay"] == "outline":
            return c["G"], None
        hist = self.ep["Ghist"][cam]
        t = self._next_step
        assert t is not None, "lag lookup without an announced step"
        if not hist:  # frame 0 itself
            assert t == 0, f"no history but step {t}"
            return c["G"], 0
        want = t - self.lag_steps
        if want < 0:  # protocol: t < lag -> hold frame 0
            h = hist[0]
            assert h[0] == 0, f"first history frame is step {h[0]}, not 0"
            return h[1], 0
        hit = [h for h in hist if h[0] == want]
        assert len(hit) == 1, f"LAG_ASSERT: no unique history frame at step {want} (t={t}, lag={self.lag_steps})"
        return hit[0][1], hit[0][0]

    def finish(self, obs, step, model_md5=None):
        if not self.armed:
            return None
        if self.cell is not None:
            assert int(step) == self._next_step, f"finish(step={step}) but armed for {self._next_step}"
        S = (self.cap or {}).get("_S")
        row = super().finish(obs, step, model_md5)
        if row is None:
            return row
        lag = None
        for v, d in (row.get("mi") or {}).items():
            if "lag_step" in d and self.mi["overlay"] == "outline_lag":
                want = max(int(step) - self.lag_steps, 0)
                assert d["lag_step"] == want, f"LAG_ASSERT: took step {d['lag_step']}, expected {want}"
                lag = d["lag_step"]
        row["mi_lag_ok"] = lag is not None if self.mi and self.mi["overlay"] == "outline_lag" else None
        sign = float(getattr(self.benv, "sign", 1.0))
        if S is None:
            row["mi_S"] = {"error": "no armed observable update"}
            self.ep.setdefault("S_pre", []).append(None)  # review 11-08 item 9: never fall back to an older frame
        else:
            pre, post = S
            fix = dict(pre)
            if "robot0_eef_quat" in fix:
                fix["robot0_eef_quat"] = fix["robot0_eef_quat"] * sign
            row["mi_S"] = {"pre_eq_post": all(np.array_equal(pre[k], post[k]) for k in pre),
                           "obs_eq_pre": all(np.array_equal(np.asarray(obs[k]), fix[k]) for k in fix),
                           "keys": sorted(pre)}
            self.ep.setdefault("S_pre", []).append(fix)
        self._next_step = None
        return row

    def pre_edit_state(self):
        """The (BodyEnv-fixed) pre-edit state observables of the last finished frame."""
        s = self.ep.get("S_pre") if self.ep else None
        return s[-1] if s else None

    def note_model_input(self, observation):
        """md5 of the 224 images actually sent (openpi request keys)."""
        md = {"agent": mr._md5(observation["observation/image"]), "wrist": mr._md5(observation["observation/wrist_image"])}
        if self.ep is not None and self.ep["rows"]:
            r, f = self.ep["rows"][-1], self.ep["frames"][-1]
            r["checks"]["model_input_md5"] = md
            r["checks"]["model_input_is_edited"] = all(md[v] == f[v]["model_md5"] for v in ("agent", "wrist") if v in f)
        return md

    def note_state(self, ok):
        if self.ep is not None and self.ep["rows"]:
            self.ep["rows"][-1].setdefault("mi_S", {})["sent_eq_pre"] = bool(ok)

    def dump_fields(self):
        out = super().dump_fields()
        if self.cell is None or self.ep is None:
            return out
        rows = self.ep["rows"]
        lagd = [(r.get("mi") or {}) for r in rows]
        act = [next((d[v]["lag_step"] for v in ("agent", "wrist") if v in d and d[v].get("lag_step") is not None), -1)
               for d in lagd]
        tgt = [max(int(r["step"]) - self.lag_steps, 0) if self.mi["overlay"] == "outline_lag" else -1 for r in rows]
        if self.mi["overlay"] != "outline_lag":
            act = [-1] * len(rows)
        fr = self.ep["frames"]
        for v in ("agent", "wrist"):  # gate item 13: md5 of the edit mask actually used, per query frame
            out[f"mi_mask_md5_{v}"] = np.asarray([mr._md5(f[v]["M"]) if v in f else "" for f in fr])
        out.update(mi_lag_step_actual=np.asarray(act, np.int32), mi_lag_step_target=np.asarray(tgt, np.int32),
                   mi_cell=self.cell, mi_query_every=np.int32(self.query_every), mi_lag_steps=np.int32(self.lag_steps),
                   mi_model_prep=self.prep,
                   mi_S_checks=json.dumps([r.get("mi_S") for r in rows]),
                   mi_lag_ok=np.asarray([bool(r.get("mi_lag_ok")) if r.get("mi_lag_ok") is not None else True
                                         for r in rows], bool))
        return out
