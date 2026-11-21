"""TOY2 amendment-1 manipulation-check geometry (protocol_toy2_amendment1.md 4.2): pinhole back-projection, the OFT
preprocessing coordinate map, free-boundary SDT displacement, gap between two masks. Pure numpy / scipy.
Raw image convention: sim.render arrays, OpenGL rows (row 0 = bottom of the view), camera looks along -z, x right, y up
(same as probe_common.project_raw, checked against segmentation shifts in toy1 / toy2 QC)."""
import numpy as np
from scipy import ndimage

RES, MODEL = 256, 224
S_CROP = np.sqrt(0.9)


def cam_params(sim, cam, res=RES):
    m, d = sim.model, sim.data
    c = m.camera_name2id(cam)
    return d.cam_xpos[c].copy(), d.cam_xmat[c].reshape(3, 3).copy(), 0.5 * res / np.tan(np.radians(m.cam_fovy[c]) / 2)


def project(cp, pts, res=RES):
    p, R, f = cp
    xc = (np.asarray(pts, float).reshape(-1, 3) - p) @ R
    return np.stack([res / 2 + f * xc[:, 1] / -xc[:, 2], res / 2 + f * xc[:, 0] / -xc[:, 2]], 1)


def backproject(cp, rows, cols, depth, res=RES):
    """raw pixels (row, col) with metric depth along the optical axis -> world points"""
    p, R, f = cp
    d = np.asarray(depth, float)
    xc = (np.asarray(cols, float) - res / 2) * d / f
    yc = (np.asarray(rows, float) - res / 2) * d / f
    pc = np.stack([xc, yc, -d], 1)
    return pc @ R.T + p


def to_model(rc):
    """raw (row, col) -> model-input (row, col): rotate 180, resize 256->224, centre crop sqrt(0.9) and resize back"""
    rc = np.asarray(rc, float)
    r = (RES - 1) - rc
    r = (r + 0.5) * MODEL / RES - 0.5
    off = (1 - S_CROP) / 2 * MODEL
    return (r + 0.5 - off) / S_CROP - 0.5


def expected_shift(cp, mask_raw, depth_raw, shift_world, uhat):
    """median displacement along uhat (model coordinates) of the visible pixels of a rigidly translated thing"""
    rr, cc = np.nonzero(mask_raw)
    if not len(rr):
        return None
    X = backproject(cp, rr, cc, depth_raw[rr, cc])
    a = to_model(project(cp, X))
    b = to_model(project(cp, X + np.asarray(shift_world)))
    return float(np.median((b - a) @ uhat))


def model_dir(cp, point, vec):
    a = to_model(project(cp, [point]))[0]
    b = to_model(project(cp, [np.asarray(point) + 0.01 * np.asarray(vec) / np.linalg.norm(vec)]))[0]
    v = b - a
    return v / max(np.linalg.norm(v), 1e-12)


def sdt(m):
    """signed distance to the mask contour (inside negative, outside positive), contour taken half-way between pixel
    centres: +-0.5 for the first pixel on either side"""
    m = np.asarray(m, bool)
    out = ndimage.distance_transform_edt(~m) - 0.5
    ins = ndimage.distance_transform_edt(m) - 0.5
    return np.where(m, -ins, out)


def free_boundary(M0, occluder, uhat, edge=2, occ_px=2, ndot=0.5):
    """free boundary points of M0 (not within `edge` px of the image border, not within `occ_px` of the occluder),
    with outward normals n (gradient of SDT_0) and |n . uhat| > ndot. Returns (points, normals, sdt0)."""
    M0 = np.asarray(M0, bool)
    if not M0.any():
        return np.zeros((0, 2), int), np.zeros((0, 2)), None
    S0 = sdt(M0)
    bnd = M0 & ~ndimage.binary_erosion(M0)
    H, W = M0.shape
    rr, cc = np.nonzero(bnd)
    keep = (rr > edge) & (rr < H - 1 - edge) & (cc > edge) & (cc < W - 1 - edge)
    if occluder is not None and np.asarray(occluder).any():
        near = ndimage.binary_dilation(np.asarray(occluder, bool), iterations=occ_px)
        keep &= ~near[rr, cc]
    rr, cc = rr[keep], cc[keep]
    gs = ndimage.gaussian_filter(S0, 1.0)
    gr, gc = np.gradient(gs)
    n = np.stack([gr[rr, cc], gc[rr, cc]], 1)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-9
    rr, cc, n = rr[ok], cc[ok], n[ok] / nn[ok, None]
    sel = np.abs(n @ uhat) > ndot
    return np.stack([rr[sel], cc[sel]], 1), n[sel], S0


def sdt_shift(pts, normals, S0, Md, uhat):
    """s = median_p[-(SDT_d(p) - SDT_0(p)) / (n_p . uhat)] (amendment 1, Astra 'freeze-before' fix)"""
    if len(pts) == 0:
        return None
    Sd = sdt(Md)
    r, c = pts[:, 0], pts[:, 1]
    return float(np.median(-(Sd[r, c] - S0[r, c]) / (normals @ uhat)))


def min_gap(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return None
    return float(ndimage.distance_transform_edt(~b)[a].min())


def occluder_raw(target_raw, depth_raw, margin=0.01, ring=3, win=7):
    """Raw pixels next to the target (within `ring` px) whose surface is NOT clearly behind it (depth < the local
    farthest target depth + margin). Boundary points of the target next to such pixels are occlusion / contact edges
    whose image position does not follow a render-only shift of the target, so they are not free boundary points."""
    T = np.asarray(target_raw, bool)
    D = np.asarray(depth_raw, float)
    far = ndimage.maximum_filter(np.where(T, D, -np.inf), size=win)
    return (~T) & ndimage.binary_dilation(T, iterations=ring) & (D < far + margin)


def from_model(rc):
    """inverse of to_model: model-input (row, col) -> raw (row, col)"""
    rc = np.asarray(rc, float)
    off = (1 - S_CROP) / 2 * MODEL
    x = (rc + 0.5) * S_CROP + off - 0.5
    x = (x + 0.5) * RES / MODEL - 0.5
    return (RES - 1) - x


def expected_shift_points(cp, pts_model, target_raw, depth_raw, shift_world, uhat, win=3):
    """median displacement along uhat (model coordinates) of the SAME free boundary points used by sdt_shift: each point is
    mapped back to the raw image, given the depth of the nearest target pixel (window `win`), back-projected, translated,
    re-projected and mapped forward again (QC 4.2(a): the boundary sits at a different depth from the interior, so the
    all-pixel median under-predicts the boundary displacement through parallax)."""
    if len(pts_model) == 0:
        return None
    T = np.asarray(target_raw, bool)
    raw = from_model(np.asarray(pts_model, float))
    H, W = T.shape
    rows, cols, ds = [], [], []
    for r, c in raw:
        ri, ci = int(round(r)), int(round(c))
        r0, r1, c0, c1 = max(ri - win, 0), min(ri + win + 1, H), max(ci - win, 0), min(ci + win + 1, W)
        sub = T[r0:r1, c0:c1]
        if not sub.any():
            continue
        yy, xx = np.nonzero(sub)
        k = np.argmin((yy + r0 - r) ** 2 + (xx + c0 - c) ** 2)
        rows.append(r); cols.append(c); ds.append(depth_raw[yy[k] + r0, xx[k] + c0])
    if not rows:
        return None
    X = backproject(cp, np.array(rows), np.array(cols), np.array(ds))
    a = to_model(project(cp, X))
    b = to_model(project(cp, X + np.asarray(shift_world)))
    return float(np.median((b - a) @ uhat))
