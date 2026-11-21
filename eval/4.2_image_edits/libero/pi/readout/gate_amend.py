"""maskinfo-pi gate amendment (deviation 2, 2026-11-12, main session; written and run BEFORE any outcome of the formal
blocks was read; never reads `success`: every npz goes through the frozen gate's NoSuccessNpz).

The frozen gate (tools/gate_maskinfo_pi.py 580ec5b8, output out/gate_maskinfo_pi.json) marks keys bad for two reasons
that turned out to be artifacts of the checking code, not of the data:
  item 7 "scene != OFT C0": exact float equality of step-0 qpos between the openpi client and the OFT client.
  item 1 "mi_gl_renderer not an object": zoo_client_mi.py records the in-process renderer once per id(env); when a later
         task's env object reuses the id of a freed one, the assert is skipped and the field is written as null.
This script clears a key ONLY IF its bad reasons are exactly those strings AND the per-key evidence below holds:
  item 7: same init id, same joint names, same shapes, max |qpos0 - qpos0_OFT| <= 1e-12 (rad or m), reference fixtures
          identical (max diff == 0).
  item 1: npz gpu and server_gpu contain H200; the client log of the same block / cell / shard / job has >= 1
          MI_GL_RENDERER line, every one contains H200, and the log has no 'llvmpipe'; and the step-0 segmentation mask
          md5 of both views plus the step-0 mi_rows pixel counts are identical to those of another cell of the same key
          whose mi_gl_renderer is a valid H200 object (same state at step 0, so same render output => same renderer).
Everything else in the gate json is copied unchanged. Output: out/gate_maskinfo_pi_amended.json (+ evidence json)."""
import sys, os, glob, re, json, hashlib, copy
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
import gate_maskinfo_pi as G

SRC, DST, EVD = "out/gate_maskinfo_pi.json", "out/gate_maskinfo_pi_amended.json", "out/gate_amend_evidence.json"
OFT = os.path.expanduser("$PROJ_ROOT/maskinfo/out/runs")
R7, R1 = "scene != OFT C0", "mi_gl_renderer not an object"
TOL = 1e-12
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for p in (DST, EVD):
    if os.path.exists(p):
        raise SystemExit(f"{p} exists")
g = json.load(open(SRC)); out = copy.deepcopy(g); ev = []

def npz_of(b, c, key):
    t, e = map(int, re.match(r"t(\d+)_e(\d+)$", key).groups())
    fs = sorted(glob.glob(f"out/runs/{b}/{c}/t{t}_e{e}_i*.npz")); assert len(fs) == 1, (b, c, key, fs)
    return fs[0], t, e, int(re.search(r"_i(\d+)\.npz$", fs[0]).group(1))

def step0(z):
    rows = json.loads(str(z["mi_rows"][()]))
    r0 = {v: {k: rows[0][v][k] for k in ("step", "G_px", "P_full_px", "P_vis_px", "R_px", "T_full_px", "box0_px")} for v in ("agent", "wrist")}
    return (str(np.asarray(z["mi_mask_md5_agent"]).ravel()[0]), str(np.asarray(z["mi_mask_md5_wrist"]).ravel()[0]), json.dumps(r0, sort_keys=True))

def valid_glr(z):
    try:
        x = json.loads(str(z["mi_gl_renderer"][()]))
    except Exception:
        return False
    return isinstance(x, dict) and all(isinstance(x.get(v), str) and "H200" in x[v] and "llvmpipe" not in x[v].lower() for v in ("physical", "reference"))

def ok7(b, c, key):
    f, t, e, i = npz_of(b, c, key); suite = b.split("_")[1]
    fo = sorted(glob.glob(f"{OFT}/oft_{suite}/C0/t{t}_e{e}_i*.npz"))
    if len(fo) != 1: return False, {"why": "no unique OFT dump"}
    io = int(re.search(r"_i(\d+)\.npz$", fo[0]).group(1))
    z, zo = G.NoSuccessNpz(f), G.NoSuccessNpz(fo[0])
    try:
        n, no = [str(x) for x in z["joint_names"]], [str(x) for x in zo["joint_names"]]
        q, qo = np.asarray(z["qpos"][0]), np.asarray(zo["qpos"][0]); r, ro = np.asarray(z["ref_body_pos"]), np.asarray(zo["ref_body_pos"])
        if i != io or n != no or q.shape != qo.shape or r.shape != ro.shape: return False, {"why": "init / names / shape"}
        dq, dr = float(np.abs(q - qo).max()), float(np.abs(r - ro).max())
        return (dq <= TOL and dr == 0.0), {"dq_max": dq, "dr_max": dr, "init": i}
    finally:
        z.close(); zo.close()

def ok1(b, c, key, jobs):
    f, t, e, i = npz_of(b, c, key); shard = 20 if e <= 24 else 25
    logs = [p for p in glob.glob(f"logs/clients/c_{b}_{c}_e{shard}_*.log") if re.search(r"_(\d+)\.log$", p) and re.search(r"_(\d+)\.log$", p).group(1) in jobs]
    if len(logs) != 1: return False, {"why": f"client logs {logs}"}
    txt = open(logs[0], errors="replace").read(); lines = [l for l in txt.splitlines() if l.startswith("MI_GL_RENDERER")]
    if not lines or any("H200" not in l for l in lines) or "llvmpipe" in txt.lower(): return False, {"why": "client log renderer lines", "n": len(lines)}
    z = G.NoSuccessNpz(f)
    try:
        if "H200" not in str(z["gpu"][()]) or "H200" not in str(z["server_gpu"][()]): return False, {"why": "gpu field"}
        mine = step0(z)
    finally:
        z.close()
    for c2 in [f"C{k}" for k in range(9)]:
        if c2 == c: continue
        z2 = G.NoSuccessNpz(npz_of(b, c2, key)[0])
        try:
            if valid_glr(z2):
                same = step0(z2) == mine
                return same, {"ref_cell": c2, "step0_equal": same, "log": os.path.basename(logs[0]), "log_glr_lines": len(lines)}
        finally:
            z2.close()
    return False, {"why": "no reference cell with a valid renderer record"}

summary = {}
for b, gb in g["blocks"].items():
    jobs = set(map(str, gb.get("jobs") or [])); cleared = {"1": 0, "7": 0}; kept = 0
    for c, cv in gb["cells"].items():
        oc = out["blocks"][b]["cells"][c]
        for key, kv in cv["keys"].items():
            if kv["status"] == "PASS": continue
            its = kv["items"]; good = True; e = {"block": b, "cell": c, "key": key, "orig": its}
            if not set(its) <= {"1", "7"} or its.get("7", [R7]) != [R7] or its.get("1", [R1]) != [R1]:
                good = False; e["why"] = "other reasons present"
            else:
                if "7" in its:
                    k7, e["ev7"] = ok7(b, c, key); good &= k7
                if "1" in its:
                    k1, e["ev1"] = ok1(b, c, key, jobs); good &= k1
            e["cleared"] = bool(good); ev.append(e)
            if good:
                oc["keys"][key] = {"items": {}, "status": "PASS", "amended_from": its}
                for it in its: cleared[it] += 1
            else:
                kept += 1
        bad = sorted([list(map(int, re.match(r"t(\d+)_e(\d+)$", k).groups())) for k, kv in oc["keys"].items() if kv["status"] != "PASS"])
        oc["bad_keys"] = bad
        rb = {}
        for k, kv in oc["keys"].items():
            for it in kv["items"]: rb[it] = rb.get(it, 0) + 1
        oc["reasons_by_item"] = rb
    ob = out["blocks"][b]
    for it in ("1", "7"):
        n = sum(1 for cv in ob["cells"].values() for kv in cv["keys"].values() if it in kv["items"])
        ob["items"][it] = dict(ob["items"][it], bad_key_count=n, status=("PASS" if n == 0 and not ob["items"][it].get("block_reasons") else "FAIL"),
                               amended_from=g["blocks"][b]["items"][it])
    summary[b] = {"cleared_item1_cellkeys": cleared["1"], "cleared_item7_cellkeys": cleared["7"], "not_cleared_cellkeys": kept,
                  "bad_union_before": len({tuple(k) for cv in gb["cells"].values() for k in cv["bad_keys"]}),
                  "bad_union_after": len({tuple(k) for cv in ob["cells"].values() for k in cv["bad_keys"]})}
out["amendment"] = {"deviation": "protocol_maskinfo_pi_deviations.md item 2 (2026-11-12)", "source_gate_sha256": sha(SRC),
                    "amend_script_sha256": sha(os.path.abspath(__file__)), "tol_qpos": TOL, "summary": summary}
json.dump(out, open(DST, "w"), ensure_ascii=False); json.dump(ev, open(EVD, "w"), ensure_ascii=False)
dq = [e["ev7"]["dq_max"] for e in ev if "ev7" in e and "dq_max" in e["ev7"]]
print(json.dumps(summary, indent=1)); print("item7 dq_max over all checked cell-keys: max %.3e  dr_max max %.3e" % (max(dq), max(e["ev7"]["dr_max"] for e in ev if "ev7" in e and "dr_max" in e["ev7"])))
print("not cleared examples:", [(e["block"], e["cell"], e["key"], e.get("why"), e.get("ev1"), e.get("ev7")) for e in ev if not e["cleared"]][:6])
print("sha", sha(SRC)[:16], "->", sha(DST)[:16], "evidence", sha(EVD)[:16])
