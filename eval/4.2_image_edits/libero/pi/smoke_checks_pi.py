"""maskinfo-pi smoke checks on the npz dumps (blind: success is never loaded; no per-episode length or count is
printed, only booleans and the first failing reason).
Usage: python -I tools/smoke_checks_pi.py --runs out/smoke_runs [--rerun out/smoke_rerun] --lag 15 --every 5
Checks per npz: labels (mi_cell / mi_block == path, cadence, lag, replan, chunk_len), H200 client + server GPU,
mi_error null, hook self checks (returned obs == edited, same sub-step renders, edits only inside E, model input
md5 == edited image through the openpi path), proprio check (pre == post == returned obs, sent == pre-edit),
lag (C5 took step t - lag, or frame 0 while t < lag; every frame with t >= lag has iou_lag / open_changed_lag in
both views = manipulation check item 7), query cadence, noise fingerprints unique within an episode.
Across cells of a key: noise fingerprint sequences equal on their common prefix. Rerun: actions, chunks, query
images and edit masks bit-identical to the first run of the same cell and key.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--rerun", default="")
ap.add_argument("--lag", type=int, default=15)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--master", type=int, default=5478)
ap.add_argument("--state-logs", default="", help="glob of server state jsonl files (gate item 14)")
a = ap.parse_args()
NEVER = {"success"}


class Z:
    """npz accessor that refuses the outcome field."""

    def __init__(self, f):
        self.z = np.load(f, allow_pickle=True)
        self.files = set(self.z.files)

    def __getitem__(self, k):
        assert k not in NEVER, "outcome field requested"
        return self.z[k]

    def s(self, k):
        return np.asarray(self[k]).item()

    def j(self, k):
        return json.loads(str(self.s(k)))


def check_one(f, block, cell):
    z = Z(f)
    bad = []
    if str(z.s("mi_cell")) != cell or str(z.s("mr_cell")) != cell:
        bad.append("mi_cell/mr_cell != path")
    if str(z.s("mi_block")) != block:
        bad.append("mi_block != path")
    if int(z.s("mi_query_every")) != a.every or int(z.s("mi_lag_steps")) != a.lag or int(z.s("mi_replan")) != a.every:
        bad.append("cadence / lag / replan label")
    cl = int(z.s("mi_chunk_len"))
    if cl not in (10, 50):
        bad.append(f"chunk_len {cl}")
    if "H200" not in str(z.s("gpu")) or "H200" not in str(z.s("server_gpu")):
        bad.append("not H200 (client or server)")
    glr = z.j("mi_gl_renderer") if "mi_gl_renderer" in z.files else None
    if not glr or any(v is not None and ("H200" not in v or "llvmpipe" in v.lower()) for v in glr.values()) \
            or glr.get("physical") is None:
        bad.append(f"in-process GL_RENDERER {glr}")
    if z.j("mi_error") is not None:
        bad.append(f"episode error {z.j('mi_error')}")
    spec = z.j("mi_spec")
    if spec.get("cell") != cell or spec.get("query_every") != a.every or spec.get("lag_steps") != a.lag \
            or spec.get("model_prep") != "openpi":
        bad.append("spec fields")
    if cell == "C8" and not (spec.get("region") == "seg" and spec.get("fill") == "bg"):
        bad.append("C8 spec is not seg + bg")
    steps = np.asarray(z["mr_steps"]).astype(int)
    if (steps % a.every).any() or steps[0] != 0 or (np.diff(steps) != a.every).any():
        bad.append("frame steps not 0, every, 2*every, ...")
    qs = np.asarray(z["query_step"]).astype(int)
    if not set(qs.tolist()) <= set(steps.tolist()) or (np.diff(qs) != a.every).any() or qs[0] != 0:
        bad.append("query steps")
    chk = z.j("mr_checks")
    for r in chk:
        for v in ("agent", "wrist"):
            c = r.get(v, {})
            if c.get("same_substep") is not True or c.get("final_eq_edited") is not True:
                bad.append(f"self check {v}")
                break
    if any(r.get("model_input_is_edited") is False for r in chk[: len(qs)]) or \
            sum(1 for r in chk if r.get("model_input_is_edited") is True) != len(qs):
        bad.append("model input md5 != edited image (openpi path)")
    rows = z.j("mi_rows")
    for r, st in zip(rows, steps):
        for v in ("agent", "wrist"):
            d = r.get(v)
            if d is None:
                bad.append(f"no mi row {v}")
                continue
            if d.get("out_E_diff_px"):
                bad.append(f"pixels changed outside E ({v})")
            if st >= a.lag and cell in ("C4", "C5") and ("iou_lag" not in d or "open_changed_lag" not in d):
                bad.append(f"item 7: no lag frame for step >= lag ({v})")
            if cell == "C5":
                if d.get("lag_step") != max(int(st) - a.lag, 0):
                    bad.append(f"C5 took step {d.get('lag_step')} not t - lag ({v})")
    if cell == "C5" and not np.asarray(z["mi_lag_ok"]).all():
        bad.append("mi_lag_ok false")
    S = z.j("mi_S_checks")
    if not all(s and s.get("pre_eq_post") and s.get("obs_eq_pre") for s in S):
        bad.append("proprio pre/post/obs mismatch")
    if not all(s.get("sent_eq_pre") for s in S[: len(qs)]) or not np.asarray(z["mi_S_sent_ok"]).all():
        bad.append("sent state != pre-edit state")
    if "mi_S_server_eq_sent" not in z.files or not bool(z.s("mi_S_server_eq_sent")):
        bad.append("server-recorded first-request state missing or != sent")
    import hashlib
    ms, mt, su = int(z.s("mi_noise_master")), str(z.s("mi_model_tag")), str(z.s("mi_suite")).replace("libero_", "")
    tt, ee = int(z.s("task")), int(z.s("ep"))
    want = [int.from_bytes(hashlib.sha256(f"{ms}|{mt}|{su}|{tt}|{ee}|{q}".encode()).digest()[:8], "big") % 2**63
            for q in range(len(qs))]
    if [int(x) for x in np.asarray(z["noise_seed"])] != want or ms != a.master:
        bad.append("noise_seed not recomputable from (MASTER, model, suite, task, ep, q)")
    la, lt = np.asarray(z["mi_lag_step_actual"]), np.asarray(z["mi_lag_step_target"])
    if cell == "C5" and (len(la) != len(steps) or (la != lt).any()):
        bad.append("mi_lag_step_actual != target")
    nm = [str(x) for x in np.asarray(z["mi_noise_md5"])]
    if len(set(nm)) != len(nm) or len(nm) != len(qs):
        bad.append("noise fingerprints not unique per query")
    for k in ("mi_cell", "mi_block", "action", "mi_obj_pos", "query_chunk", "mr_img_agent", "mr_img_wrist"):
        if k not in z.files:
            bad.append(f"missing {k}")
    return bad, nm, z, [str(x) for x in np.asarray(z["mi_mask_md5_agent"])] + [str(x) for x in np.asarray(z["mi_mask_md5_wrist"])]


def walk(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "C*", "t*_e*_i*.npz"))):
        block, cell = f.split(os.sep)[-3], f.split(os.sep)[-2]
        m = re.fullmatch(r"t(\d+)_e(\d+)_i(\d+)\.npz", os.path.basename(f))
        out[(block, cell, int(m.group(1)), int(m.group(2)))] = f
    return out


first = walk(a.runs)
n_bad = 0
noise, masks = {}, {}
srv = {}
if a.state_logs:
    for sf in glob.glob(a.state_logs):
        for line in open(sf):
            r = json.loads(line)
            srv.setdefault(int(r["noise_seed"]), []).append(r["state"])
for (block, cell, t, e), f in sorted(first.items()):
    bad, nm, zz, mk = check_one(f, block, cell)
    masks[(block, t, e, cell)] = mk
    if a.state_logs:
        s0 = int(np.asarray(zz["noise_seed"])[0])
        sent = np.asarray(zz["mi_S_server_received"], np.float64)
        if s0 not in srv or not any(np.array_equal(np.asarray(x, np.float64), sent) for x in srv[s0]):
            bad.append("server state log has no matching first-request state")
    noise[(block, t, e, cell)] = nm
    n_bad += bool(bad)
    print(f"NPZ {block}/{cell}/t{t}_e{e}: {'OK' if not bad else 'BAD ' + '; '.join(sorted(set(bad)))}")
# common random numbers: same key, different cells -> same noise sequence on the common prefix
for (block, t, e) in sorted({(k[0], k[1], k[2]) for k in noise}):
    seqs = {c: s for (b, tt, ee, c), s in noise.items() if (b, tt, ee) == (block, t, e)}
    ref = seqs[sorted(seqs)[0]]
    same = all(s[: min(len(s), len(ref))] == ref[: min(len(s), len(ref))] for s in seqs.values())
    n_bad += not same
    print(f"NOISE {block} t{t}_e{e}: cells={','.join(sorted(seqs))} same_on_common_prefix={same}")
# gate item 13: C8 uses the same mask as C1, per frame (common prefix; masks only, not changed pixels)
for (block, t, e) in sorted({(k[0], k[1], k[2]) for k in masks}):
    m1, m8 = masks.get((block, t, e, "C1")), masks.get((block, t, e, "C8"))
    if m1 is not None and m8 is not None:
        n = min(len(m1), len(m8)) // 2
        h1, h8 = m1[: len(m1) // 2][:n] + m1[len(m1) // 2:][:n], m8[: len(m8) // 2][:n] + m8[len(m8) // 2:][:n]
        print(f"C8_MASK {block} t{t}_e{e}: frame0_equal={h1[0] == h8[0]} (closed loop: later frames differ by trajectory)")
# other blocks share nothing: different model tag or suite -> different seeds (sanity only)
blocks = sorted({k[0] for k in noise})
if len(blocks) > 1:
    q0 = {b: [s[0] for (bb, *_), s in noise.items() if bb == b][0] for b in blocks}
    print("NOISE first-query fingerprints differ across blocks:", len(set(q0.values())) == len(q0))
if a.rerun:
    second = walk(a.rerun)
    for k, f2 in sorted(second.items()):
        f1 = first.get(k)
        if f1 is None:
            print(f"RERUN {k}: no first run"); n_bad += 1
            continue
        z1, z2 = Z(f1), Z(f2)
        eq = {x: (np.asarray(z1[x]).shape == np.asarray(z2[x]).shape and np.array_equal(np.asarray(z1[x]), np.asarray(z2[x])))
              for x in ("action", "query_chunk", "mr_img_agent", "mr_img_wrist", "mr_M_agent", "mr_M_wrist", "qpos",
                        "mi_noise_md5")}
        ok = all(eq.values())
        n_bad += not ok
        print(f"RERUN {k[0]}/{k[1]}/t{k[2]}_e{k[3]}: bit_identical={ok} {'' if ok else eq}")
print("SMOKE_CHECKS_BAD", n_bad)
