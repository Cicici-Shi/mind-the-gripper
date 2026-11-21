"""maskrepl: observation-hook masking for protocol_maskrepl.md (sha256 564ba794...), section 8. 2026-11-03.

One MaskHook per BodyEnv (physical env only; body Panda, S family). It wraps the robosuite env's
_update_observables (the place LIBERO-VPro's a3_blur.py hooks) and the two camera sensors:

  * Only on observations the policy reads (the eval loop calls arm() before the env.step whose observation feeds a
    query, and before set_init_state) the hook does anything; otherwise every call is passed through untouched.
    Non-query observations are never read by OFT and each control step re-renders its camera images from scratch,
    so editing only the query observations gives the policy exactly the inputs of "edit every step".
  * Inside each camera sensor call (the simulation sub-step at which robosuite samples the camera) it renders, in
    the same sub-step state and back to back: the RGB (the original sensor; for alpha cells with the 8 gripper geoms
    at alpha 0), the MuJoCo segmentation (decoded from the raw ID colours in int64, geoms only), and the alpha-0
    "real background" RGB + segmentation (never sent to the policy). sim.data.time and md5(qpos) are recorded for
    each of the three renders (protocol 2, gate 4.2).
  * After the wrapped _update_observables returns it edits the observable value and _obs_cache (as a3_blur.py does).
    Fill schedules follow their source:
      - Trace fills (bg / black / mosaic): applied ONCE to the clean sampled image with the region mask, like
        vla_trace/evaluation/libero.py applies apply_image_mask_to_obs_inplace to the step observation.
      - VPro blur: applied at EVERY _update_observables call from the sampling call to the end of the control step,
        each time to the current (already blurred) observable value with the box projected at that call's
        sub-step pose: this is literally what a3_blur.py's update_observables_with_world_blur does (the camera is
        sampled once per control step, the hook runs after every one of the ~25 sub-step updates).
      - box region with a Trace fill (X1, X3): the region is the union of the boxes projected at those same calls
        (the area VPro's repeated application covers), filled once.
  * finish(obs) after the step: checks the returned observation equals the edited image, computes the section 5
    quantities (256 raw space and 224 model-input space) and stores the dump fields.

Ported code (line-by-line, names kept):
  Trace: vla_trace/behavior/patchmask.py, github.com/VLA-Trace/VLA-Trace commit
    62a04134536548fe68b339cabd7379a6d6fb00a3, file sha256 89ed938fb6d2e6d382f3146cc63cf8c77746e14ab513be1270bd0c0756442c74
    (vendor/trace_patchmask_62a04134.py): apply_patch_mask, _mask_bool_2d, _fill_with_ring_background,
    _mosaic_region, _dilate_mask; gripper instance = first entry of model.instances_to_ids whose name contains
    "Gripper"/"gripper" (_ensure_segmentation_ids), pixel value = index + 1 (robosuite instance sensor, as in
    mask_robot_exc_gripper). NOTE: with stock LIBERO SegmentationRenderEnv the public mask_gripper path raises
    (get_segmentation_instances has no "gripper" key); the instance definition above is the only one consistent
    with the rest of the file and with tests/test_behavior_stage3.py's fake env.
  VPro: source/A3_interaction_part_occlusion/a3_blur.py, github.com/huiqiongli/LIBERO-VPro-Dataset commit
    df1e151a2e4ea2aab987252e385d51646acc39d4, file sha256 8c37beee4c61ed5eb65f53f7e1c2d55fc1b9fb8f3b399161ed5dfd88bb2f696a
    (MIT; vendor/vpro_a3_blur_df1e151a.py): _convex_hull, _camera_projection, _project, _gripper_world_rectangles,
    _blur_image (gaussian path); spec A3_blur_gripper_candidate.json (sha256 6a4eb048...).
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROTOCOL_SHA256 = "564ba79421c08439297cea248e7f5e9c58672d85a71afeffbc5575c198925ab2"
CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))
SHORT = {"agentview": "agent", "robot0_eye_in_hand": "wrist"}
TRACE = {"mask_value": 0, "bg_ring_width": 8, "mosaic_block": 8, "fallback_value": 0,
         "source": "VLA-Trace patchmask.py @62a04134536548fe68b339cabd7379a6d6fb00a3 sha256 89ed938f"}
VPRO = {"gripper_patch_half_extents_m": [0.054, 0.015, 0.030], "gripper_patch_axis_offset_m": -0.0175,
        "gripper_patch_local_offset_m": [0.0, 0.0, 0.0], "gaussian_radius_fraction": 0.12,
        "edge_feather_fraction": 0.002, "observation_image_convention": "opengl",
        "mask_to_source_body_visible_pixels": False, "grip_site": "gripper0_grip_site",
        "source": "LIBERO-VPro a3_blur.py @df1e151a sha256 8c37beee, spec A3_blur_gripper_candidate.json sha256 6a4eb048"}
# cell -> (region, fill, views); protocol section 2
CELLS = {
    "S_hook": ("none", "none", "both"),
    "R1": ("seg", "bg", "both"), "R2": ("seg", "black", "both"), "R3": ("seg", "mosaic", "both"),
    "R4": ("box", "blur", "both"),
    "X1": ("box", "black", "both"), "X2": ("seg", "blur", "both"), "X3": ("box", "bg", "both"),
    "V1": ("seg", "bg", "agent"), "V2": ("seg", "bg", "wrist"),
    "V3": ("box", "blur", "agent"), "V4": ("box", "blur", "wrist"),
    "O1": ("alpha", "none", "agent"), "O2": ("alpha", "none", "wrist"),
}
VIEWS = {"both": ("agentview", "robot0_eye_in_hand"), "agent": ("agentview",), "wrist": ("robot0_eye_in_hand",)}
TRACE_MODE = {"bg": "background_fill", "black": "black", "mosaic": "mosaic"}
SHADOW_RING = 12   # protocol 5: D(G, 12)
LF_BLOCK = 8       # protocol 5: 8x downsample
BLUR_BAND = 3      # protocol 4.2: ceil(3 * 0.002 * 256) + 1


def mask_spec(cell):
    region, fill, views = CELLS[cell]
    return {"cell": cell, "region": region, "fill": fill, "views": views, "trace": TRACE, "vpro": VPRO,
            "protocol_sha256": PROTOCOL_SHA256,
            "fill_schedule": {"blur": "every _update_observables call from the camera sampling call to the end of the "
                                      "control step (a3_blur.py hook)",
                              "bg/black/mosaic": "once on the sampled image (Trace)"},
            "box_region": "union of the boxes projected at those calls (sampling call box = box0)"}


# ---------------------------------------------------------------- Trace port (patchmask.py @62a04134), verbatim logic
def apply_patch_mask(image, mask, *, mode, mask_value=0, bg_ring_width=8, mosaic_block=8):
    """Return an image with the selected region masked."""
    if mode not in ("none", "black", "background_fill", "mosaic"):
        raise ValueError(f"Unsupported patchmask mode: {mode}")
    output = np.asarray(image).copy()
    mask2d = _mask_bool_2d(mask, output.shape[:2])
    if mode == "none" or not mask2d.any():
        return output
    if mode == "black":
        output[mask2d] = np.asarray(mask_value, dtype=output.dtype)
        return output
    if mode == "background_fill":
        _fill_with_ring_background(output, mask2d, ring_width=bg_ring_width, fallback_value=mask_value)
        return output
    if mode == "mosaic":
        _mosaic_region(output, mask2d, block=mosaic_block)
        return output
    raise ValueError(f"Unsupported patchmask mode: {mode}")


def _mask_bool_2d(mask, shape):
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
        cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
        array = array[np.ix_(rows, cols)]
    return array.astype(bool)


def _fill_with_ring_background(image, mask, *, ring_width, fallback_value):
    ring = _dilate_mask(mask, int(ring_width)) & ~mask
    source = ring if ring.any() else ~mask
    if source.any():
        fill = image[source].reshape(-1, image.shape[-1]).mean(axis=0)
        if image.dtype == np.uint8:
            fill = np.clip(np.round(fill), 0, 255).astype(np.uint8)
        else:
            fill = fill.astype(image.dtype)
        image[mask] = fill
    else:
        image[mask] = np.asarray(fallback_value, dtype=image.dtype)


def _mosaic_region(image, mask, *, block):
    block = max(int(block), 1)
    height, width, channels = image.shape
    for y0 in range(0, height, block):
        y1 = min(y0 + block, height)
        for x0 in range(0, width, block):
            x1 = min(x0 + block, width)
            submask = mask[y0:y1, x0:x1]
            if not submask.any():
                continue
            patch = image[y0:y1, x0:x1].copy()
            fill = patch.reshape(-1, channels).mean(axis=0)
            if image.dtype == np.uint8:
                fill = np.clip(np.round(fill), 0, 255).astype(np.uint8)
            else:
                fill = fill.astype(image.dtype)
            patch[submask] = fill
            image[y0:y1, x0:x1] = patch


def _dilate_mask(mask, radius):
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    side = 2 * radius + 1
    for dy in range(side):
        for dx in range(side):
            out |= padded[dy: dy + height, dx: dx + width]
    return out


def trace_gripper_instance_index(instance_names):
    """_ensure_segmentation_ids gripper branch: first instance whose name contains Gripper/gripper."""
    for idx, name in enumerate(instance_names):
        if any(piece in name for piece in ("Gripper", "gripper")):
            return idx
    return None


# ---------------------------------------------------------------- VPro port (a3_blur.py @df1e151a)
def convex_hull(points):
    """Monotone-chain hull returned as (column, row) coordinates."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return unique

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def camera_projection(sim, camera_name, height, width):
    camera_id = int(sim.model.camera_name2id(camera_name))
    position = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    corrected_rotation = rotation @ np.diag((1.0, -1.0, -1.0))
    world_to_camera_rotation = corrected_rotation.T
    fovy = float(sim.model.cam_fovy[camera_id])
    focal = 0.5 * height / np.tan(fovy * np.pi / 360.0)
    return position, world_to_camera_rotation, float(focal)


def project(sim, world_points, camera_name, height, width, convention="opengl"):
    camera_position, world_to_camera_rotation, focal = camera_projection(sim, camera_name, height, width)
    camera_points = (world_points - camera_position) @ world_to_camera_rotation.T
    depth = camera_points[:, 2]
    valid = depth > 1.0e-5
    if np.count_nonzero(valid) < 3:
        return [], depth.tolist()
    points = camera_points[valid]
    columns = focal * points[:, 0] / points[:, 2] + width / 2.0
    rows = focal * points[:, 1] / points[:, 2] + height / 2.0
    if convention == "opengl":
        rows = (height - 1.0) - rows
    polygon = convex_hull(np.column_stack((columns, rows)))
    return polygon, depth.tolist()


def gripper_box_corners(sim, grip_site_id, half=VPRO["gripper_patch_half_extents_m"],
                        axis_offset=VPRO["gripper_patch_axis_offset_m"], local_offset=VPRO["gripper_patch_local_offset_m"]):
    rotation = np.asarray(sim.data.site_xmat[grip_site_id], dtype=np.float64).reshape(3, 3)
    grip_point = np.asarray(sim.data.site_xpos[grip_site_id], dtype=np.float64)
    center = grip_point + rotation[:, 2] * float(axis_offset) + rotation @ np.asarray(local_offset, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return np.asarray([center + rotation @ np.asarray((sx * half[0], sy * half[1], sz * half[2]))
                       for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=np.float64)


def blur_image(image, polygons, radius, edge_feather, mask_bool=None):
    """a3_blur.py _blur_image, gaussian operator, allowed_mask None. mask_bool (X2, seg region) replaces the drawn
    polygons by a raster mask (same feather / blur / composite)."""
    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if mask_bool is None:
        mask = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)
    else:
        mask = Image.fromarray(np.where(np.asarray(mask_bool, bool), 255, 0).astype(np.uint8), mode="L")
    if edge_feather > 0.0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_feather))
    mask_array = np.asarray(mask)
    mask_pixels = int(np.count_nonzero(mask_array))
    if mask_pixels == 0:
        return np.asarray(image).copy(), 0
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    result = Image.composite(blurred, source, mask)
    return np.asarray(result, dtype=np.asarray(image).dtype), mask_pixels


def raster_polygon(polygon, height, width):
    mask = Image.new("L", (width, height), 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return np.asarray(mask) > 0


# ---------------------------------------------------------------- section 5 helpers
def _dilate_sq(mask, r):
    """Square dilation (Chebyshev radius r), separable; equals patchmask._dilate_mask(mask, r)."""
    m = np.asarray(mask, bool)
    if r <= 0:
        return m.copy()
    a = m.copy()
    for s in range(1, r + 1):
        a[s:] |= m[:-s]
        a[:-s] |= m[s:]
    b = a.copy()
    for s in range(1, r + 1):
        b[:, s:] |= a[:, :-s]
        b[:, :-s] |= a[:, s:]
    return b


def _boundary(mask):
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    p = np.pad(m, 1, constant_values=False)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def _gradmag(img):
    g = np.asarray(img, np.float64).mean(axis=2)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def _nanmean(a):
    a = np.asarray(a, np.float64)
    return float(a.mean()) if a.size else None


def section5(I_edit, I_bg, M, G, T_full, T_vis, block=LF_BLOCK, ring=SHADOW_RING, M_hides=True):
    """M_hides=False for alpha-0 views: the region shows the real background, so it hides no target pixel (occ 0)."""
    H, W = G.shape
    absd = np.abs(np.asarray(I_edit, np.int16) - np.asarray(I_bg, np.int16)).mean(axis=2)
    out = {"area": float(M.sum()) / (H * W), "G_px": int(G.sum()), "M_px": int(M.sum()),
           "T_full_px": int(T_full.sum()), "T_vis_px": int(T_vis.sum())}
    Mo = M if M_hides else np.zeros_like(M)
    out["occ"] = None if T_full.sum() == 0 else float(1.0 - (T_vis & ~Mo).sum() / T_full.sum())
    out["res_in"] = _nanmean(absd[M]) if M.any() else None
    dG = _boundary(G)
    if dG.any():
        ge, gb = _gradmag(I_edit)[dG].mean(), _gradmag(I_bg)[dG].mean()
        out["edge"] = None if gb == 0 else float(ge / gb)
    else:
        out["edge"] = None
    D = _dilate_sq(G, ring)
    sh = D & ~M
    out["shadow"] = _nanmean(absd[sh]) if sh.any() else None
    if G.any():
        hb, wb = H // block, W // block
        e = np.asarray(I_edit, np.float64)[:hb * block, :wb * block].reshape(hb, block, wb, block, 3).mean(axis=(1, 3))
        b = np.asarray(I_bg, np.float64)[:hb * block, :wb * block].reshape(hb, block, wb, block, 3).mean(axis=(1, 3))
        Db = D[:hb * block, :wb * block].reshape(hb, block, wb, block).any(axis=(1, 3))
        out["lf"] = float(np.abs(e - b).mean(axis=2)[Db].mean()) if Db.any() else None
    else:
        out["lf"] = None
    return out


_PREP = {}


def _oft_prep():
    if not _PREP:
        from experiments.robot.openvla_utils import center_crop_image, resize_image_for_policy
        _PREP["resize"], _PREP["crop"] = resize_image_for_policy, center_crop_image
    return _PREP["resize"], _PREP["crop"]


def to_model_input(img):
    """Exact OFT image path: 180 deg rotation (get_libero_image), resize_image_for_policy(224) (JPEG + lanczos3) ->
    returned as r (what run_libero_eval hands to get_action), then prepare_images_for_vla's centre crop
    (Image.fromarray(r).convert("RGB") -> center_crop_image) -> returned as c (what the backbone sees)."""
    resize, crop = _oft_prep()
    r = np.asarray(resize(np.ascontiguousarray(np.asarray(img)[::-1, ::-1]), 224), dtype=np.uint8)
    c = crop(Image.fromarray(r).convert("RGB"))
    c = np.asarray(c.convert("RGB") if hasattr(c, "convert") else c, dtype=np.uint8)
    return c, r


def mask_to_model(mask, size=224, crop_area=0.9):
    """Geometric twin of the image path for masks: rotate 180, bilinear resize to 224, centre crop of area 0.9,
    bilinear resize back, threshold 0.5 (masks have no JPEG step)."""
    m = Image.fromarray(np.where(np.asarray(mask, bool)[::-1, ::-1], 255, 0).astype(np.uint8), mode="L")
    m = m.resize((size, size), Image.BILINEAR)
    side = size * np.sqrt(crop_area)
    o = (size - side) / 2.0
    m = m.resize((size, size), Image.BILINEAR, box=(o, o, o + side, o + side))
    return np.asarray(m) >= 128


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- hook
class MaskHook:
    def __init__(self, benv, cell, res=256, metrics_path=None, log=print):
        self.benv, self.res, self.log = benv, res, log
        self.rs = benv.env.env  # robosuite env (persists across hard resets)
        self._orig_update = self.rs._update_observables
        self.rs._update_observables = self._update
        self.metrics_path = metrics_path
        self.set_cell(cell)
        self.armed, self.cap, self.ep = False, None, None
        self.task = self.epi = None

    def set_cell(self, cell):
        self.cell = cell
        if cell is None:  # pass-through (plain S)
            self.region, self.fill, self.views = "none", "none", ()
            self.spec = None
            return
        self.region, self.fill, v = CELLS[cell]
        self.views = VIEWS[v]
        self.spec = mask_spec(cell)

    # -- episode bookkeeping
    def begin_episode(self, task, ep):
        """Call after env.reset() (the model may have been rebuilt)."""
        self.task, self.epi = task, ep
        sim = self.rs.sim
        m = sim.model
        palm, fingers = self.benv._grip_parts(m)
        hide = [g for g in palm + fingers if m.geom_rgba[g][3] > 0]
        inst_names = list(self.rs.model.instances_to_ids.keys())
        name2id = {inst: i for i, inst in enumerate(inst_names)}
        lut = np.zeros(m.ngeom + 1, np.int32)  # index geom+1 (geom -1 -> 0)
        for g, inst in self.rs.model.geom_ids_to_instances.items():
            lut[int(g) + 1] = name2id[inst] + 1
        gidx = trace_gripper_instance_index(inst_names)
        targets = list(getattr(self.rs, "obj_of_interest", []) or [])
        tgeoms = [g for g, inst in self.rs.model.geom_ids_to_instances.items() if inst in targets]
        gr = self.rs.robots[0].gripper
        gr = gr.get("right", next(iter(gr.values()))) if isinstance(gr, dict) else gr
        fingers_vis = [int(m.geom_name2id(n)) for n in gr.visual_geoms if "finger" in str(n).lower()]
        body_grip = [g for g in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0_")]
        self.ep = {"sim_id": id(sim), "hide": hide, "hide_names": [m.geom_id2name(g) or f"geom{g}" for g in hide],
                   "inst_lut": lut, "gripper_inst": None if gidx is None else gidx + 1,
                   "gripper_inst_name": None if gidx is None else inst_names[gidx], "targets": targets,
                   "target_geoms": np.asarray(sorted(int(g) for g in tgeoms)), "body_grip": np.asarray(body_grip), "fingers_vis": np.asarray(fingers_vis),
                   "grip_site": int(m.site_name2id(VPRO["grip_site"])), "frames": [], "rows": []}
        if gidx is None:
            raise RuntimeError("maskrepl: no gripper instance in model.instances_to_ids")

    def arm(self):
        if self.cell is not None:
            self.armed, self.cap = True, {}

    # -- rendering helpers (same GL context as the sensors)
    def _state(self):
        d = self.rs.sim.data
        return [float(d.time), hashlib.md5(np.asarray(d.qpos).tobytes()).hexdigest()]

    def _seg(self, cam):
        sim = self.rs.sim
        m = sim.model
        ctx = sim._render_context_offscreen
        W = H = self.res
        # Sites are hidden in RGB (robosuite negative alpha) but the segmentation pass ignores alpha and draws them
        # (gripper0_grip_site_cylinder etc. cover gripper pixels). Trace's env built with camera_segmentations shrinks
        # every site for exactly this reason (robot_env.py); we hide all site groups for the segmentation pass only.
        sg = np.array(ctx.vopt.sitegroup, copy=True)
        ctx.vopt.sitegroup[:] = 0
        try:
            ctx.render(width=W, height=H, camera_id=m.camera_name2id(cam), segmentation=True)
            raw = ctx.read_pixels(W, H, segmentation=False).astype(np.int64)
        finally:
            ctx.vopt.sitegroup[:] = sg
        sid = raw[:, :, 0] + raw[:, :, 1] * 256 + raw[:, :, 2] * 65536  # segid + 1, 0 = background
        lut = np.full(ctx.scn.ngeom + 2, -1, dtype=np.int64)
        for i in range(ctx.scn.ngeom):
            gg = ctx.scn.geoms[i]
            if gg.segid != -1:
                lut[gg.segid + 1] = gg.objid if int(gg.objtype) == 5 else -2  # -2: non-geom object (site, ...)
        sid[(sid < 0) | (sid >= len(lut))] = 0
        return lut[sid]

    def _alpha0(self):
        m = self.rs.sim.model
        h = self.ep["hide"]
        old = m.geom_rgba[h, 3].copy()
        m.geom_rgba[h, 3] = 0.0
        return old

    def _restore(self, old):
        self.rs.sim.model.geom_rgba[self.ep["hide"], 3] = old

    def _make_sensor(self, orig, cam):
        def wrapped(obs_cache):
            if not self.armed:
                return orig(obs_cache)
            assert self.ep is not None and self.ep["sim_id"] == id(self.rs.sim), "begin_episode() not called after reset"
            st = {"rgb": self._state()}
            old = self._alpha0() if (self.region == "alpha" and cam in self.views) else None
            try:
                img = orig(obs_cache)
            finally:
                if old is not None:
                    self._restore(old)
            st["seg"] = self._state()
            gid = self._seg(cam)
            st["bg"] = self._state()
            old = self._alpha0()
            try:
                bg = np.asarray(self.rs.sim.render(camera_name=cam, width=self.res, height=self.res, depth=False))
                bg_gid = self._seg(cam)
            finally:
                self._restore(old)
            st["end"] = self._state()
            self.cap[cam] = {"clean": np.array(img, copy=True), "gid": gid, "bg": np.array(bg, copy=True),
                             "bg_gid": bg_gid, "st": st, "fresh": True, "nsample": self.cap.get(cam, {}).get("nsample", 0) + 1}
            return img
        wrapped._mr_cam = cam
        wrapped.__modality__ = orig.__modality__  # robosuite @sensor attribute (Observable.modality reads it)
        return wrapped

    def _ensure_sensors(self):
        for key, cam in CAMS:
            ob = self.rs._observables.get(key)
            if ob is not None and getattr(ob._sensor, "_mr_cam", None) is None:
                ob._sensor = self._make_sensor(ob._sensor, cam)

    def _update(self, force=False):
        if self.armed:
            self._ensure_sensors()
        self._orig_update(force=force)
        if self.armed and self.cap:
            self._post_call()

    def _post_call(self):
        sim = self.rs.sim
        H = W = self.res
        radius = max(2.0, min(H, W) * VPRO["gaussian_radius_fraction"])
        feather = min(H, W) * VPRO["edge_feather_fraction"]
        for key, cam in CAMS:
            c = self.cap.get(cam)
            if c is None:
                continue
            if c["fresh"]:
                c["fresh"] = False
                c["calls"] = 0
                inst = self.ep["inst_lut"][c["gid"] + 1]
                inst[c["gid"] < 0] = 0
                c["G"] = inst == self.ep["gripper_inst"]
                c["G_body"] = np.isin(c["gid"], self.ep["body_grip"])
                c["F"] = np.isin(c["gid"], self.ep["fingers_vis"])  # a3_blur.py source geoms (two finger visuals)
                poly0, _ = project(sim, gripper_box_corners(sim, self.ep["grip_site"]), cam, H, W)
                c["box0"] = raster_polygon(poly0, H, W)
                c["union"] = np.zeros((H, W), bool)
                c["edited"] = None
            c["calls"] += 1
            if cam not in self.views or self.region in ("none", "alpha"):
                continue
            ob = self.rs._observables[key]
            poly = None
            if self.region == "box":
                poly, _ = project(sim, gripper_box_corners(sim, self.ep["grip_site"]), cam, H, W)
                grown = raster_polygon(poly, H, W) & ~c["union"]
                c["union"] |= grown
            M = c["G"] if self.region == "seg" else c["union"]
            if self.fill == "blur":
                cur = np.asarray(ob.obs)
                if self.region == "box":
                    new, _ = blur_image(cur, [poly], radius, feather)
                else:
                    new, _ = blur_image(cur, [], radius, feather, mask_bool=M)
            else:
                if c["edited"] is not None and (self.region == "seg" or not grown.any()):
                    continue
                new = apply_patch_mask(c["clean"], M, mode=TRACE_MODE[self.fill], mask_value=TRACE["mask_value"],
                                       bg_ring_width=TRACE["bg_ring_width"], mosaic_block=TRACE["mosaic_block"])
            ob._current_observed_value = new
            self.rs._obs_cache[key] = new.copy()
            c["edited"] = new

    # -- after the step / set_init_state whose observation feeds a query
    def finish(self, obs, step, model_md5=None):
        if not self.armed:
            return None
        self.armed = False
        cap, self.cap = self.cap, None
        H = W = self.res
        row = {"cell": self.cell, "task": self.task, "ep": self.epi, "step": int(step), "views": {}, "checks": {}}
        frame = {"step": int(step)}
        for key, cam in CAMS:
            v = SHORT[cam]
            c = cap.get(cam)
            if c is None or "G" not in c:
                row["checks"][v] = {"error": "camera not sampled under the armed step"}
                continue
            final = np.asarray(obs[key])
            edited = c["edited"] if c["edited"] is not None else c["clean"]
            edited_view = cam in self.views and self.region not in ("none",)
            if self.region == "alpha" or not edited_view:
                M = c["G"] if (self.region == "alpha" and edited_view) else np.zeros((H, W), bool)
            else:
                M = c["G"] if self.region == "seg" else c["union"]
            tg = self.ep["target_geoms"]
            T_full = np.isin(c["bg_gid"], tg)
            T_vis = T_full if (self.region == "alpha" and edited_view) else np.isin(c["gid"], tg)
            G = c["G"]
            st = c["st"]
            # identity outside the edit region (fill cells: outside M; blur: outside M dilated by 3; others: all)
            diff = np.any(final != c["clean"], axis=2)
            if edited_view and self.region in ("seg", "box"):
                band = _dilate_sq(M, BLUR_BAND) if self.fill == "blur" else M
                outside = ~band
            else:
                band, outside = np.zeros((H, W), bool), np.ones((H, W), bool)
            chk = {"final_eq_edited": bool(np.array_equal(final, edited)),
                   "out_diff_px": int((diff & outside).sum()),
                   "band_diff_px": int((diff & band & ~M).sum()),
                   "band_maxdiff": int(np.abs(final.astype(np.int16) - c["clean"].astype(np.int16))[band & ~M].max())
                   if (band & ~M).any() else 0,
                   "in_changed_px": int((diff & M).sum()),
                   "same_substep": st["rgb"] == st["seg"] == st["bg"] == st["end"],
                   "t_rgb": st["rgb"][0], "q_rgb": st["rgb"][1][:12], "t_seg": st["seg"][0], "q_seg": st["seg"][1][:12],
                   "t_bg": st["bg"][0], "q_bg": st["bg"][1][:12],
                   "nsample": c["nsample"], "calls": c["calls"],
                   "G_inst_px": int(G.sum()), "G_body_px": int(c["G_body"].sum()),
                   "inst_body_xor": int((G ^ c["G_body"]).sum()), "nongeom_px": int((c["gid"] == -2).sum()),
                   "bg_grip_px": int(np.isin(c["bg_gid"], self.ep["body_grip"]).sum()),
                   "box0_frac": float(c["box0"].sum()) / (H * W),
                   "box0_cover_G": None if not G.any() else float((G & c["box0"]).sum() / G.sum()),
                   "union_frac": float(c["union"].sum()) / (H * W),
                   "F_px": int(c["F"].sum()),
                   "box0_cover_fingers": None if not c["F"].any() else float((c["F"] & c["box0"]).sum() / c["F"].sum())}
            row["checks"][v] = chk
            hides = not (self.region == "alpha" and edited_view)
            q256 = section5(final, c["bg"], M, G, T_full, T_vis, M_hides=hides)
            fe, fe_r = to_model_input(final)
            be, _ = to_model_input(c["bg"])
            q224 = section5(fe, be, mask_to_model(M), mask_to_model(G), mask_to_model(T_full), mask_to_model(T_vis),
                            M_hides=hides)
            q256["q224"] = q224
            row["views"][v] = q256
            frame[v] = {"img": final.copy(), "G": np.packbits(G), "M": np.packbits(M), "model_md5": _md5(fe_r)}
            if getattr(self, "keep_debug", False):  # gate tool only
                frame[v].update(clean=c["clean"].copy(), bg=c["bg"].copy(), Gm=G.copy(), Mm=M.copy())
        if model_md5 is not None:
            row["checks"]["model_input_md5"] = model_md5
        self.ep["rows"].append(row)
        self.ep["frames"].append(frame)
        return row

    def note_model_input(self, observation):
        """md5 of the 224 images handed to get_action (before OFT's centre crop)."""
        md = {"agent": _md5(observation["full_image"]), "wrist": _md5(observation["wrist_image"])}
        if self.ep is not None and self.ep["rows"]:
            r, f = self.ep["rows"][-1], self.ep["frames"][-1]
            r["checks"]["model_input_md5"] = md
            r["checks"]["model_input_is_edited"] = all(md[v] == f[v]["model_md5"] for v in ("agent", "wrist") if v in f)
        return md

    def end_episode(self):
        """Append this episode's per-query rows to the metrics jsonl (after the queries filled in model md5)."""
        if self.metrics_path and self.ep is not None and self.ep["rows"]:
            with open(self.metrics_path, "a") as f:
                for r in self.ep["rows"]:
                    f.write(json.dumps(r) + "\n")

    def dump_fields(self):
        if self.ep is None or self.cell is None:
            return {}
        fr = self.ep["frames"]
        out = {"mr_spec": json.dumps(self.spec), "mr_cell": self.cell,
               "mr_hidden_geoms": json.dumps(self.ep["hide_names"] if self.region == "alpha" else []),
               "mr_bg_hidden_geoms": json.dumps(self.ep["hide_names"]),
               "mr_gripper_instance": self.ep["gripper_inst_name"], "mr_targets": json.dumps(self.ep["targets"]),
               "mr_steps": np.asarray([f["step"] for f in fr], np.int32),
               "mr_checks": json.dumps([r["checks"] for r in self.ep["rows"]])}
        for v in ("agent", "wrist"):
            ok = [f for f in fr if v in f]
            if ok:
                out[f"mr_img_{v}"] = np.stack([f[v]["img"] for f in ok]).astype(np.uint8)
                out[f"mr_G_{v}"] = np.stack([f[v]["G"] for f in ok])
                out[f"mr_M_{v}"] = np.stack([f[v]["M"] for f in ok])
                out[f"mr_model_md5_{v}"] = np.asarray([f[v]["model_md5"] for f in ok])
        return out
