"""maskinfo: the eight cells of protocol_maskinfo.md (C0 .. C7) on top of the FROZEN maskrepl hook, 2026-11-06.

embody/maskrepl.py is a byte copy of $PROJ_ROOT/maskrepl/embody/maskrepl.py (sha256 dffe5b04...) and is NOT modified:
C0 / C1 / C2 / C3 run exactly the maskrepl code paths (S_hook; seg x black = R2; box x black = X1; alpha 0 on BOTH
views = the O1 / O2 alpha switch with views="both"). The new cells only add branches after the frozen _post_call:

  C4  alpha0 + correct outline   : I_bg, O(t)      \\ P_full(t) set to (0,0,0)
  C5  alpha0 + lagged outline    : I_bg, O(t - L)  \\ P_full(t) set to (0,0,0)   (L = lag_steps, same view, same episode)
  C6  box-black, protected       : C2 image, P_vis(t) restored to I
  C7  ectopic box (v3 2.3)       : I, M7(t) set to (0,0,0); M7 = mirrored (+ shifted) box union, clipped to the frame,
                                   >= 0.8 |B|, avoiding G u P_full u R

O(t) = (G \\ erode4^k_in(G)) u (dilate8^k_out(G) \\ G), k_in = w // 2, k_out = w - k_in (w = outline_width; protocol v3
fixes w = 3 = 1 inner + 2 outer, O = dG u (dilate8^2(G) \\ G)); the erosion treats pixels beyond the frame as G
(protocol v3 2.1), so no line is drawn along the image edge; section 5's dG keeps the maskrepl definition. Protected / avoided set P (protocol v2 2.0, Astra ruling 5),
configurable (PROTECT_SETS): "interactive" (default) = obj_of_interest u every movable object (LIBERO objects_dict)
u articulated fixture parts (fixture bodies with a joint on their chain below the fixture root: drawers, doors);
NOT table / floor / walls (arena geoms, no instance) nor fixed furniture bodies (cabinet carcass, stove);
"targets" = obj_of_interest only (protocol v1). P is a GEOM set (articulated parts are sub-bodies of one
segmentation instance). P_full = P pixels in the alpha-0 segmentation (gripper absent), P_vis = P pixels in the
visible segmentation. The same P is used by C4, C5, C6 and C7. With MI_PSET=<json> the enumerated P names of the
task must equal the frozen list (tools/mi_cells.sh) or the episode raises.

Per query frame the hook stores, per view, row["mi"][view] (outline pixel counts, protected truncation, lag frame,
C7 candidate / overlap / truncation, P / G / R pixel counts, the expected edit region E and the pixels changed outside
E relative to the cell's own clean render). Section 5 quantities of the frozen hook are recomputed with M = E for
C6 and C7 (their occlusion must count the restored / added pixels; see protocol 2.4 item 4).
"""
import hashlib
import json
import os

import numpy as np

from embody import maskrepl as mr
from embody.maskrepl import (CAMS, SHORT, MaskHook, _dilate_sq, apply_patch_mask, gripper_box_corners, mask_to_model,
                             project, raster_polygon, section5, to_model_input)

QUERY_EVERY = 8
OPEN_THR = 0.039 / 2  # protocol 2.4 item 7: half of the Panda full opening (finger joint 1)
# cell -> frozen-hook (region, fill, views) + new branches
MI_CELLS = {
    "C0": {"region": "none", "fill": "none", "overlay": "none", "protect": "none", "bgmask": "none", "maskrepl": "S_hook"},
    "C1": {"region": "seg", "fill": "black", "overlay": "none", "protect": "none", "bgmask": "none", "maskrepl": "R2"},
    "C2": {"region": "box", "fill": "black", "overlay": "none", "protect": "none", "bgmask": "none", "maskrepl": "X1"},
    "C3": {"region": "alpha", "fill": "alpha0", "overlay": "none", "protect": "none", "bgmask": "none",
           "maskrepl": "O1+O2 alpha switch, both views (= toy1 S_hid)"},
    "C4": {"region": "alpha", "fill": "alpha0", "overlay": "outline", "protect": "pfull", "bgmask": "none", "maskrepl": None},
    "C5": {"region": "alpha", "fill": "alpha0", "overlay": "outline_lag", "protect": "pfull", "bgmask": "none",
           "maskrepl": None},
    "C6": {"region": "box", "fill": "black", "overlay": "none", "protect": "pvis", "bgmask": "none", "maskrepl": None},
    "C7": {"region": "none", "fill": "none", "overlay": "none", "protect": "none", "bgmask": "box_mirror", "maskrepl": None},
}
PROTECT_SETS = ("interactive", "targets")
# robot instances (arm, mount, gripper) = instances whose geoms belong to bodies with these prefixes; LIBERO has no
# segmentation instance for table / floor / walls (arena geoms), so "interactive" = every other instance
ROBOT_BODY_PREFIX = ("robot0_", "gripper0_", "mount0_")
# C7 candidates after the mirror, (dx, dy) in POLICY orientation (+x right, +y down; the raw observation is that
# image rotated 180 deg, so the raw shift is (-dx, -dy)); protocol 2.3 order
C7_SHIFTS = [(0, 0), (32, 0), (-32, 0), (0, 32), (0, -32), (64, 0), (-64, 0), (0, 64), (0, -64),
             (32, 32), (32, -32), (-32, 32), (-32, -32)]
C7_MAX_OVERLAP = 0.05
C7_MIN_AREA = 0.8  # protocol v3 2.3: a candidate qualifies only if its in-frame area >= 0.8 |B|


def protocol_sha():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocol_maskinfo.md")
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def mi_spec(cell, lag_steps, outline_width, protect_set):
    s = dict(MI_CELLS[cell])
    s.update(cell=cell, views="both", lag_steps=int(lag_steps), outline_width=int(outline_width),
             outline_color=[0, 0, 0], protect_set=protect_set, c7_shifts_policy_xy=C7_SHIFTS,
             c7_max_overlap=C7_MAX_OVERLAP, c7_min_area_frac=C7_MIN_AREA, query_every=QUERY_EVERY, protocol="protocol_maskinfo.md",
             protocol_sha256=protocol_sha(), maskrepl_sha256="dffe5b0420685a544c84510fbd7f5755c0ce694bf88358ba2249f4ca5c93b257",
             trace=mr.TRACE, vpro=mr.VPRO)
    return s


# ---------------------------------------------------------------- mask helpers (new code)
def erode4(mask, k=1, outside=False):
    """4-neighbour erosion; outside = value of the pixels beyond the frame (protocol v3 2.1: True for the outline, so
    G touching the image edge draws no line along the edge)."""
    m = np.asarray(mask, bool)
    for _ in range(k):
        p = np.pad(m, 1, constant_values=outside)
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def outline(G, width=2):
    """O = (G minus its k_in-fold 4-neighbour erosion) u (k_out-fold 3x3 dilation minus G)."""
    G = np.asarray(G, bool)
    k_in = width // 2  # w = 2: 1 + 1 (protocol 2.1); w = 3: 1 inner + 2 outer (2 + 1 collapses the dG edge metric)
    k_out = width - k_in
    inner = G & ~erode4(G, k_in, outside=True) if k_in > 0 else np.zeros_like(G)
    outer = _dilate_sq(G, k_out) & ~G if k_out > 0 else np.zeros_like(G)
    return inner | outer


def shift_mask(m, dy, dx):
    """Shift a raw-array mask by (dy rows, dx cols); pixels leaving the frame are dropped."""
    H, W = m.shape
    out = np.zeros_like(m)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    if (ys.stop - ys.start) > 0 and (xs.stop - xs.start) > 0:
        out[yd, xd] = m[ys, xs]
    return out


def c7_place(B, avoid, view):
    """Protocol v3 2.3: mirror of B about the image midline (agentview left-right, wrist up-down), then the candidate
    shifts; every candidate is clipped to the frame and qualifies only if |M7| >= 0.8 |B|; overlap o = |M7 n A| / |M7|
    (in-frame area). First qualified candidate with o <= 0.05, else the qualified one with the least o (candidate 0,
    the plain mirror, always qualifies). Returns (M7, cand, overlap, trunc_px, qualified_count)."""
    B = np.asarray(B, bool)
    nB = int(B.sum())
    mir = B[:, ::-1] if view == "agent" else B[::-1, :]
    best, nq = None, 0
    for i, (dx, dy) in enumerate(C7_SHIFTS):
        M = shift_mask(mir, -dy, -dx)  # policy (dx, dy) -> raw (-dx, -dy)
        n = int(M.sum())
        if n == 0 or n < C7_MIN_AREA * nB:
            continue
        nq += 1
        o = float((M & avoid).sum()) / n
        if best is None or o < best[2]:
            best = (M, i, o)
        if o <= C7_MAX_OVERLAP:
            best = (M, i, o)
            break
    if best is None:  # empty B: nothing to place
        return np.zeros_like(B), -1, None, 0, 0
    M, i, o = best
    return M, i, o, nB - int(M.sum()), nq


# ---------------------------------------------------------------- hook
class MaskInfoHook(MaskHook):
    def __init__(self, benv, cell, res=256, metrics_path=None, log=print, lag_steps=16, outline_width=2,
                 protect_set="interactive"):
        assert lag_steps % QUERY_EVERY == 0 and lag_steps > 0, lag_steps
        assert protect_set in PROTECT_SETS, protect_set
        self.lag_steps, self.outline_width, self.protect_set = int(lag_steps), int(outline_width), protect_set
        self.det_check = False
        super().__init__(benv, cell, res=res, metrics_path=metrics_path, log=log)

    def set_cell(self, cell):
        self.cell = cell
        self.mi = None
        if cell is None:
            self.region, self.fill, self.views, self.spec = "none", "none", (), None
            return
        c = MI_CELLS[cell]
        self.mi = c
        self.region = c["region"]
        self.fill = {"alpha0": "none"}.get(c["fill"], c["fill"])
        self.views = mr.VIEWS["both"]
        self.spec = mi_spec(cell, self.lag_steps, self.outline_width, self.protect_set)

    def begin_episode(self, task, ep):
        super().begin_episode(task, ep)
        rs, m = self.rs, self.rs.sim.model
        names = list(rs.model.instances_to_ids.keys())
        n2i = {n: i + 1 for i, n in enumerate(names)}
        robot_inst = set()
        for g, inst in rs.model.geom_ids_to_instances.items():
            if (m.body_id2name(int(m.geom_bodyid[int(g)])) or "").startswith(ROBOT_BODY_PREFIX):
                robot_inst.add(inst)
        robot = [n for n in names if n in robot_inst and n != self.ep["gripper_inst_name"]]
        pset, pgeoms = self._p_set(rs, m)
        self.ep["P_geoms"] = np.asarray(sorted(pgeoms), np.int64)
        if self.spec is not None:
            self.spec = dict(self.spec, P_names=list(pset), task_key=self._task_key())
        lst = os.environ.get("MI_PSET")
        if lst:
            want = json.load(open(lst)).get(self._task_key())
            assert want is not None, f"MI_PSET has no entry for {self._task_key()}"
            assert sorted(want) == sorted(pset), f"P list mismatch {sorted(pset)} != frozen {sorted(want)}"
        self.ep.update(protect_names=pset,
                       robot_names=robot, robot_ids=np.asarray([n2i[n] for n in robot], np.int32),
                       instance_names=names, Ghist={cam: [] for _, cam in CAMS}, mi_rows=[])

    def _task_key(self):
        """bddl file name of the LIBERO problem (unique per task within and across the two suites)."""
        return os.path.splitext(os.path.basename(str(self.rs.bddl_file_name)))[0]

    def _p_set(self, rs, m):
        """P names (instances and articulated part bodies) and P geom ids."""
        g2i = {int(g): inst for g, inst in rs.model.geom_ids_to_instances.items()}
        names, geoms = [], set()
        self.ep["P_name_geoms"] = byname = {}
        if self.protect_set == "targets":
            keep = set(self.ep["targets"])
            names = sorted(keep)
            geoms = {g for g, i in g2i.items() if i in keep}
            for g, i in g2i.items():
                if i in keep:
                    byname.setdefault(i, []).append(g)
            return names, geoms
        objs = set(getattr(rs, "objects_dict", {}).keys()) | set(self.ep["targets"])
        fixtures = getattr(rs, "fixtures_dict", {})
        for g, inst in g2i.items():
            if inst in objs:
                geoms.add(g)
                byname.setdefault(inst, []).append(g)
        names += sorted(i for i in set(g2i.values()) if i in objs)
        for fname, fx in fixtures.items():
            if fname in objs:
                continue
            root = m.body_name2id(fx.root_body)
            parts = set()
            for g, inst in g2i.items():
                if inst != fname:
                    continue
                b = int(m.geom_bodyid[g])
                chain = []
                while b != root and b > 0:
                    chain.append(b)
                    b = int(m.body_parentid[b])
                art = [c for c in chain if m.body_jntnum[c] > 0]
                if art:
                    geoms.add(g)
                    parts.add(m.body_id2name(art[-1]))
                    byname.setdefault(m.body_id2name(art[-1]), []).append(g)
            names += sorted(parts)
        return names, geoms

    def _inst(self, gid):
        inst = self.ep["inst_lut"][gid + 1]
        inst[gid < 0] = 0
        return inst

    # ---- same-state double render (protocol 4 item 1), only when det_check is on
    def _make_sensor(self, orig, cam):
        w = super()._make_sensor(orig, cam)

        def w2(obs_cache):
            img = w(obs_cache)
            if self.armed and self.det_check and self.cap and cam in self.cap:
                self._det(cam)
            return img
        w2._mr_cam = cam
        w2.__modality__ = w.__modality__
        return w2

    def _det(self, cam):
        c = self.cap[cam]
        sim = self.rs.sim
        st0 = self._state()
        a = np.asarray(sim.render(camera_name=cam, width=self.res, height=self.res, depth=False))
        b = np.asarray(sim.render(camera_name=cam, width=self.res, height=self.res, depth=False))
        s2 = self._seg(cam)
        old = self._alpha0()
        try:
            bg2 = np.asarray(sim.render(camera_name=cam, width=self.res, height=self.res, depth=False))
            bs2 = self._seg(cam)
        finally:
            self._restore(old)

        def rd(x, y):
            d = np.abs(np.asarray(x, np.int16) - np.asarray(y, np.int16)).max(axis=2)
            return [int((d > 0).sum()), int(d.max())]
        c["det"] = {"same_state": st0 == self._state() == c["st"]["end"],
                    "rgb_twice": rd(a, b), "bg_twice": rd(c["bg"], bg2),
                    "seg_twice_px": int((s2 != c["gid"]).sum()), "bgseg_twice_px": int((bs2 != c["bg_gid"]).sum()),
                    "sensor_vs_render": rd(c["clean"], bg2 if self.region == "alpha" else a)}

    # ---- new branches after the frozen _post_call
    def _post_call(self):
        super()._post_call()
        if self.mi is None:
            return
        for key, cam in CAMS:
            c = self.cap.get(cam)
            if c is None or "G" not in c:
                continue
            if "P_full" not in c:
                c["P_full"] = np.isin(c["bg_gid"], self.ep["P_geoms"])
                c["P_vis"] = np.isin(c["gid"], self.ep["P_geoms"])
                c["R"] = np.isin(self._inst(c["gid"]), self.ep["robot_ids"])
            new = None
            if self.mi["bgmask"] != "none":  # C7: the box union the frozen code builds only for region "box"
                self._c7_grow(cam, c)
            if self.mi["overlay"] != "none" and c.get("mi_E") is None:
                Gl, lag_step = self._lag_G(cam, c)
                O = outline(Gl, self.outline_width)
                draw = O & ~c["P_full"]
                new = c["clean"].copy()
                new[draw] = 0
                c.update(mi_O=O, mi_E=draw, mi_lag_step=lag_step)
            elif self.mi["protect"] == "pvis" and c.get("edited") is not None and c["edited"] is not c.get("mi_seen"):
                new = c["edited"].copy()
                new[c["P_vis"]] = c["clean"][c["P_vis"]]
                c["mi_E"] = c["union"] & ~c["P_vis"]
            elif self.mi["bgmask"] != "none" and c.get("mi_grew"):
                c["mi_grew"] = False
                v = SHORT[cam]
                M7, cand, o, trunc, nq = c7_place(c["union"], c["G"] | c["P_full"] | c["R"], v)
                new = c["clean"].copy()
                new[M7] = 0
                c.update(mi_E=M7, mi_c7=(cand, o, trunc, nq))
            if new is not None:
                ob = self.rs._observables[key]
                ob._current_observed_value = new
                self.rs._obs_cache[key] = new.copy()
                c["edited"] = new
                c["mi_seen"] = new

    def _c7_grow(self, cam, c):
        """Same union as the frozen code builds for region 'box' (boxes projected at every call from the sampling
        call on); the frozen code skips it for region 'none'."""
        sim = self.rs.sim
        poly, _ = project(sim, gripper_box_corners(sim, self.ep["grip_site"]), cam, self.res, self.res)
        grown = raster_polygon(poly, self.res, self.res) & ~c["union"]
        if grown.any():
            c["union"] |= grown
            c["mi_grew"] = True

    def _lag_G(self, cam, c):
        if self.mi["overlay"] == "outline":
            return c["G"], None
        hist = self.ep["Ghist"][cam]
        q = len(hist)  # index of the current query frame
        j = q - self.lag_steps // QUERY_EVERY
        if q == 0:
            return c["G"], 0  # frame 0: hold
        j = max(j, 0)
        return hist[j][1], hist[j][0]

    def finish(self, obs, step, model_md5=None):
        if not self.armed:
            return None
        cap = self.cap
        row = super().finish(obs, step, model_md5)
        if self.cell is None or row is None:
            return row
        H = W = self.res
        frame = self.ep["frames"][-1]
        mi = {}
        for key, cam in CAMS:
            v = SHORT[cam]
            c = cap.get(cam)
            if c is None or "G" not in c:
                continue
            gq = np.asarray(obs["robot0_gripper_qpos"]) if "robot0_gripper_qpos" in obs else None
            is_open = None if gq is None else bool(gq[0] > OPEN_THR)
            O_now = outline(c["G"], self.outline_width)
            prev = [h for h in self.ep["Ghist"][cam] if h[0] == int(step) - self.lag_steps]
            self.ep["Ghist"][cam].append((int(step), c["G"].copy(), is_open))
            final = np.asarray(obs[key])
            E = c.get("mi_E")
            if E is None:  # C0..C3: the frozen region
                if self.region == "seg":
                    E = c["G"]
                elif self.region == "box":
                    E = c["union"]
                else:
                    E = np.zeros((H, W), bool)
            diff = np.any(final != c["clean"], axis=2)
            G = c["G"]
            d = {"step": int(step), "E_px": int(E.sum()), "out_E_diff_px": int((diff & ~E).sum()),
                 "in_E_changed_px": int((diff & E).sum()), "G_px": int(G.sum()),
                 "P_full_px": int(c["P_full"].sum()), "P_vis_px": int(c["P_vis"].sum()), "R_px": int(c["R"].sum()),
                 "B_px": int(c["union"].sum()), "box0_px": int(c["box0"].sum()),
                 "T_full_px": int(np.isin(c["bg_gid"], self.ep["target_geoms"]).sum()),
                 "T_vis_px": int(np.isin(c["gid"], self.ep["target_geoms"]).sum()),
                 "dG_px": int(mr._boundary(G).sum()), "dG_adj_Pvis_px": int((mr._boundary(G) & _dilate_sq(c["P_vis"], 1)).sum()),
                 "open": is_open}
            if prev:
                Ol = outline(prev[0][1], self.outline_width)
                u = int((O_now | Ol).sum())
                d.update(iou_lag=None if u == 0 else float((O_now & Ol).sum()) / u,
                         open_changed_lag=None if (is_open is None or prev[0][2] is None) else bool(is_open != prev[0][2]))
            if "mi_O" in c:
                O = c["mi_O"]
                d.update(O_px=int(O.sum()), O_drawn_px=int(c["mi_E"].sum()), O_protected_px=int((O & c["P_full"]).sum()),
                         lag_step=c["mi_lag_step"], lag_clamped=bool(self.mi["overlay"] == "outline_lag" and
                                                                    int(step) - self.lag_steps < 0))
            if "mi_c7" in c:
                cand, o, trunc, nq = c["mi_c7"]
                nB = int(c["union"].sum())
                d.update(c7_cand=int(cand), c7_overlap=None if o is None else float(o), c7_trunc_px=int(trunc),
                         M7_px=int(c["mi_E"].sum()), c7_qualified=int(nq),
                         M7_over_B=None if nB == 0 else float(c["mi_E"].sum()) / nB)
            if "det" in c:
                d["det"] = c["det"]
            mi[v] = d
            # section 5 with the cell's real edit region for C6 / C7 (occlusion counts the restored / added pixels)
            if self.mi["protect"] == "pvis" or self.mi["bgmask"] != "none":
                tg = self.ep["target_geoms"]
                T_full, T_vis = np.isin(c["bg_gid"], tg), np.isin(c["gid"], tg)
                q256 = section5(final, c["bg"], E, G, T_full, T_vis, M_hides=True)
                fe, _ = to_model_input(final)
                be, _ = to_model_input(c["bg"])
                q256["q224"] = section5(fe, be, mask_to_model(E), mask_to_model(G), mask_to_model(T_full),
                                        mask_to_model(T_vis), M_hides=True)
                row["views"][v] = q256
            if v in frame:
                frame[v]["M"] = np.packbits(E)
        row["mi"] = mi
        self.ep["mi_rows"].append(mi)
        return row

    def dump_fields(self):
        out = super().dump_fields()
        if self.cell is None or self.ep is None:
            return out
        out.update(mi_spec=json.dumps(self.spec), mi_rows=json.dumps(self.ep["mi_rows"]),
                   mi_protect_names=json.dumps(self.ep["protect_names"]), mi_robot_names=json.dumps(self.ep["robot_names"]))
        return out
