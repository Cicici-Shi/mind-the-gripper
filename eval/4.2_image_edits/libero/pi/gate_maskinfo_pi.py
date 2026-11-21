"""maskinfo-pi gates (protocol_maskinfo_pi.md v2 section 4, 15 items) -> gate json. NEVER reads `success`.

Copied from the frozen $PROJ_ROOT/maskinfo/tools/gate_maskinfo.py (that file is not modified) and changed for the openpi
blocks pi05_spatial / pi05_object / pi0_spatial / pi0_object, cells C0..C8, keys = tasks x eps (formal: 10 x 20..29,
shards 20-24 / 25-29; --tasks / --eps let the same code run on the smoke keys, e.g. --tasks 0 --eps 40).

Rules that differ from the frozen gate on purpose:
  * every npz is read through NoSuccessNpz, which refuses the `success` member (it is never decompressed);
  * a missing field is a FAIL of the item that needs it -- there is no `z.get(k, default)` anywhere
    (deviations file item 10 (c)); mi_cell == mr_cell == path cell and mi_block == path block are required;
  * config lines (MI_SERVER_CFG / MI_CFG / MI_GL_RENDERER) are parsed as JSON objects that may be folded over
    several lines (deviations file item 8), never by a single-line grep.

Item -> data source (what this script checks itself; anything not listed is reported under not_checked):
  1  render: npz gpu / server_gpu contain H200; npz mi_gl_renderer {physical, reference} contain H200 and not
     llvmpipe; client-log MI_GL_RENDERER lines likewise; same-state render determinism on h200 from --manip.
  2  card lock: one card type (h200) over gpu and server_gpu of every npz of the block.
  3  first frame / same sub-step: --t0 jsonl (manip t0 mode), bad / missing keys of the suite fail the key in
     every block of that suite; frames 0, 5, 10 must be present per key.
  4  hook transparency (client part): mr_checks same_substep / final_eq_edited (both views, every frame) and
     model_input_is_edited (every queried frame); mi_rows out_E_diff_px == 0 (both views, every frame).
  S  protocol 2.2: mi_S_checks pre_eq_post / obs_eq_pre (every frame), sent_eq_pre (every queried frame),
     mi_S_sent_ok (every query).
  5  physics invariance: --manip key rows, physics_qpos_max_vs_C0 == 0 for every cell, >= 10 keys per suite.
  6  detector D: per cell list of keys (reported, never a gate failure; protocol: main readout does not drop).
  7  scene / config: align_json copy_grip_cmd == 1; mi_replan == mi_query_every == 5; query_step / mr_steps on
     the 5-step cadence from 0; mi_chunk_len (pi0 50, pi0.5 10); mi_error null; mi_model / mi_suite /
     mi_model_prep; npz mi_server_cfg == the MI_SERVER_CFG logged by the server of the same job (job from
     mi_server_cfg.state_log), its host / gpu == npz server_host / server_gpu, config / pi05 / action_horizon /
     discrete_state_input / debug_state_transform; client-log MI_CFG; step-0 qpos and reference fixtures == the
     block's C0 and == the OFT dump of the same key (--oft-ref; exact, as the frozen gate).
  8  port invariance: on-disk embody/maskinfo.py, maskrepl.py sha256 == frozen; mi_spec maskinfo_sha256 /
     maskrepl_sha256 == frozen.
  9  blind: npz mi_blind True; every record of out/runs/<block>/<cell>/blind_*.jsonl has blind == True and none
     of success / steps / queries / wall_s (nor contacts.n_steps, twin.n_paint); client EP / FINAL lines carry no
     outcome or length. A non-blind record fails the whole block.
  10 labels: mi_cell, mr_cell, mi_block present; mi_cell == mr_cell == path cell; mi_block == path block; task /
     ep in the npz == file name.
  11 lag (C5): mi_lag_step_actual == mi_lag_step_target on every frame; t0 = mr_steps[0] (must be 0, first
     action step); every frame with t >= t0 + lag takes t - lag (fraction 1.0), earlier frames hold frame 0;
     mi_lag_ok all True; mi_rows lag_step of both views == actual.
  12 noise seed: noise_seed[q] == int.from_bytes(sha256(f"{5478}|{model}|{suite}|{task}|{ep}|{q}")[:8], "big")
     % 2**63 for q = 0..n-1; mi_noise_master == 5478; mi_model_tag == block model; server logs without
     "request without noise_seed" refusals.
  13 C8 mask: C8 mi_mask_md5_agent / _wrist [0] == C1 [0] of the same key (closed loop: later frames differ).
  14 server S record: logs/server/state_<block>_<job>.jsonl has a row with noise_seed == npz noise_seed[0] whose
     state == npz mi_S_server_received element-wise; npz mi_S_server_eq_sent True.
  15 visibility: --manip-summary (tools/manip_summary_pi.py json): keys.visibility_hard_ok True, all four
     suite/view entries present; keys.rule_lag.decision == --lag (protocol 2.3 rule).
  M  the other snapshot-mode manipulation checks of 2.6 (item 1 categories incl. C8 inside G, C6 union, item 3
     border / dilate, item 4 occ, item 9 three md5 + same-moment S, hook self checks, item 2 C7) without
     violations (frozen gate: "manipulation checks not PASS" fails every block).

Usage (cwd $PROJ_ROOT/maskinfo_pi):
  python -I tools/gate_maskinfo_pi.py --protocol protocol_maskinfo_pi.md --runs out/runs --json out/gate_maskinfo_pi.json \
      --manip out/manip_pi/run_J/m_[0-9].jsonl --t0 out/manip_pi/run_J/m_*_t0.jsonl \
      --manip-summary out/manip_pi/run_J/summary.json [--logs-root logs] [--oft-ref $PROJ_ROOT/maskinfo/out/runs]
  smoke:  ... --runs out/smoke_runs_v2 --tasks 0 --eps 40 --oft-ref none
stdout: counts and PASS / FAIL only (never outcomes, episode lengths, step or query counts).
"""
import argparse, glob, hashlib, importlib.util, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

CELLS = [f"C{i}" for i in range(9)]
BLOCKS = ["pi05_spatial", "pi05_object", "pi0_spatial", "pi0_object"]
MODELS = {"pi05": {"config": "pi05_libero", "chunk": 10, "pi05": True},
          "pi0": {"config": "pi0_libero", "chunk": 50, "pi05": False}}
ALLOWED = ("h200",)  # protocol v2 4 items 1-2: h200 only
MASTER = 5478
QUERY_EVERY = 5
OUTLINE_W, PROTECT, PREP = "3", "interactive", "openpi"
FROZEN_SHA = {"maskinfo.py": "3d301badbb870c9a5491fa18a8d74136f13da2bcf4f5ff29378379b660e14653",
              "maskrepl.py": "dffe5b0420685a544c84510fbd7f5755c0ce694bf88358ba2249f4ca5c93b257"}
REC_FORBIDDEN = ("success", "steps", "queries", "wall_s")
ITEMS = ["1", "2", "3", "4", "S", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "M"]
NOT_CHECKED = {
    "1": "GL_RENDERER assert at job start (tools/gl_renderer.py in the sbatch) and 'render determinism done while "
         "the server holds the card' are job-script facts, not in the data read here",
    "4": "second half (C0 224 sent image == hook-less client on the same state, md5): no data source exists yet",
    "7": "server config identical item by item to the published pi0.5 run / models1 pi0 run (manip item 11)",
    "8": "maskinfo_pi.py overrides only the listed methods; client diff vs zoo_client_next.py only the three "
         "additions: code review, not data",
}


class NoSuccessNpz:
    """npz reader that refuses the outcome member; nothing else is hidden. Missing fields raise KeyError."""

    def __init__(self, path):
        self._z = np.load(path, allow_pickle=False)
        self.files = frozenset(n for n in self._z.files if n != "success")

    def __contains__(self, k):
        if k == "success":
            raise PermissionError("gate never reads success")
        return k in self.files

    def __getitem__(self, k):
        if k == "success":
            raise PermissionError("gate never reads success")
        if k not in self.files:
            raise KeyError(k)
        return self._z[k]

    def close(self):
        self._z.close()


def sha256f(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def card_type(gpu):
    g = str(gpu)
    if "H200" in g:
        return "h200"
    if "H100" in g:
        m = re.search(r"(\d+)\s*MiB", g)
        return "h100-96" if m and int(m.group(1)) >= 90000 else "h100-47"
    if "A100" in g and "80GB" in g:
        m = re.search(r"(\d+)\s*MiB", g)
        if m and int(m.group(1)) < 70000:
            return "a100-80-slice"
        return "a100-80"
    return "other:" + g


def tol_ok(n, mx):
    return n == 0 or (n <= 10 and mx <= 1)


def expected_seed(model, suite, task, ep, q):
    return int.from_bytes(hashlib.sha256(f"{MASTER}|{model}|{suite}|{task}|{ep}|{q}".encode()).digest()[:8], "big") % 2 ** 63


def scalar(z, k):
    a = z[k]
    if a.ndim != 0:
        raise ValueError(f"{k} not 0-d")
    return a.item()


def jsonl(files):
    for f in files:
        for l in open(f):
            if l.strip():
                yield json.loads(l)


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


_DEC = json.JSONDecoder()


def json_lines(text, tag):
    """Every `<tag> {json}` object in a log, tolerating folded lines (deviations item 8): decode in place; if that
    fails, re-join the following physical lines (without and with a space) until the object decodes."""
    out = []
    for m in re.finditer(re.escape(tag) + r"\s*(?=\{)", text):
        rest = text[m.end():]
        try:
            out.append((_DEC.raw_decode(rest)[0], "line"))
            continue
        except ValueError:
            pass
        lines, got = rest.split("\n"), None
        for n in range(2, min(len(lines), 400) + 1):
            for sep in ("", " "):
                try:
                    got = _DEC.raw_decode(sep.join(x.rstrip("\r") for x in lines[:n]))[0]
                    break
                except ValueError:
                    pass
            if got is not None:
                break
        out.append((got, "joined" if got is not None else "unparsed"))
    return out


# ------------------------------------------------------------------ manipulation-check outputs
def _rows_with_suite(files):
    rows = []
    for f in files:
        su = None
        for r in jsonl([f]):
            if r.get("header"):
                su = r["suite"]
            r["_suite"] = su
            rows.append(r)
    return rows


def summarize_manip(files):
    """Snapshot-mode jsonl of tools/manip_maskinfo_pi.py (frozen summarize_manip extended to C8, openpi item 9
    (three md5) + item 9_S, physics per suite)."""
    rows = _rows_with_suite(files)
    keys = [r for r in rows if r.get("key_row") and not r.get("t0_mode")]
    fr = [r for r in rows if "view" in r and not r.get("t0_mode")]
    out = {"n_keys": len(keys), "n_frame_views": len(fr), "hosts_gpus": sorted({(r["host"], r["gpu"]) for r in keys})}
    det = defaultdict(lambda: {"frames": 0, "seg_fail": 0, "rgb_fail": 0})
    gpu_of = {(r["_suite"], r["task"], r["ep"]): card_type(r["gpu"]) for r in keys}
    for r in fr:
        ct = gpu_of.get((r["_suite"], r["task"], r["ep"]), "unknown")
        for c, d in (r.get("det") or {}).items():
            if not d:
                continue
            x = det[ct]
            x["frames"] += 1
            x["seg_fail"] += int(d["seg_twice_px"] != 0 or d["bgseg_twice_px"] != 0)
            x["rgb_fail"] += int(not (tol_ok(*d["rgb_twice"]) and tol_ok(*d["bg_twice"])))
    out["render_det"] = {ct: dict(x, status="PASS" if x["frames"] > 0 and x["seg_fail"] == 0
                                  and x["rgb_fail"] <= 0.01 * x["frames"] else "FAIL") for ct, x in det.items()}
    phys = {}
    for su in sorted({r["_suite"] for r in keys}):
        ph = []
        for r in keys:
            if r["_suite"] != su:
                continue
            vals = r["physics_qpos_max_vs_C0"]
            missing = [c for c in CELLS[1:] if c not in vals]
            ph.append(float("inf") if missing else max(float("inf") if v is None else float(v) for v in vals.values()))
        phys[su] = {"keys": len(ph), "max": max(ph) if ph else None,
                    "status": "PASS" if len(ph) >= 10 and max(ph) == 0.0 else "FAIL"}
    out["physics"] = phys
    bad = Counter()
    for r in fr:
        for c, d in r["item1"].items():
            if isinstance(d, dict) and "ok" in d and not d["ok"]:
                bad["item1:" + c] += 1
            if isinstance(d, dict) and d.get("drawn_all_black") is False:
                bad["item1:drawn_not_black:" + c] += 1
        if r["item1"].get("C6_union_eq_C2_union") is False:
            bad["item1:C6_union"] += 1
        for c, d in r.get("item3", {}).items():
            if isinstance(d, dict) and d.get("outside_dilate8_px"):
                bad["item3:" + c] += 1
        for nm in ("item4_occ", "item4_occP"):
            occ = r.get(nm, {})
            for c in ("C3", "C4", "C5"):
                if occ.get(c) not in (None, 0.0):
                    bad[f"{nm}:{c}!=0"] += 1
            if nm == "item4_occP" and occ.get("C6") != occ.get("C1"):
                bad[f"{nm}:C6!=C1"] += 1
        for c in CELLS:  # openpi item 9: hook 224 == client 224 == independent 224 (protocol 2.2), and 9_S
            if r["item9"].get(c) is not True:
                bad["item9_md5:" + c] += 1
            if r["item9_S"].get(c) is not True:
                bad["item9_S:" + c] += 1
        for c in CELLS:  # hook self checks: every cell present, missing value = violation (no fallback)
            s = r["self"].get(c)
            if not isinstance(s, dict) or s.get("out_E_diff_px") != 0:
                bad["self_outside_E:" + c] += 1
            if not isinstance(s, dict) or s.get("same_substep") is not True:
                bad["same_substep:" + c] += 1
            if not isinstance(s, dict) or s.get("final_eq_edited") is not True:
                bad["final_eq_edited:" + c] += 1
        it2 = r.get("item2", {})
        if it2.get("B_eq_M7_plus_trunc") is False:
            bad["item2:B!=M7+trunc"] += 1
        if it2.get("M7_ge_08B") is False:
            bad["item2:M7<0.8B"] += 1
        for c in ("C4_border", "C5_border"):
            if r.get("item3", {}).get(c, {}).get("ok") is False:
                bad["item3:" + c] += 1
    out["violations"] = dict(bad)
    out["status"] = "PASS" if (keys and not bad and phys and all(v["status"] == "PASS" for v in phys.values())) else "FAIL"
    return out


def summarize_t0(files):
    rows = _rows_with_suite(files)
    keys = [r for r in rows if r.get("key_row") and r.get("t0_mode")]
    fr = [r for r in rows if "view" in r and r.get("t0_mode")]
    cells = {c: {"bad_keys": set(), "tolerated": 0} for c in CELLS}
    for r in fr:
        k = (r["_suite"], r["task"], r["ep"])
        for c, d in r["item1"].items():
            if not isinstance(d, dict) or "ok" not in d:
                continue
            cc = c.split("_")[0]
            if cc in cells:
                if not d["ok"]:
                    cells[cc]["bad_keys"].add(k)
                elif d.get("out_n", d.get("n", 0)):
                    cells[cc]["tolerated"] += 1
        for c, s in r["self"].items():
            if s.get("same_substep") is not True or s.get("final_eq_edited") is not True or s.get("out_E_diff_px"):
                cells[c]["bad_keys"].add(k)
    frames_bad = sorted([r["_suite"], r["task"], r["ep"]] for r in keys if not {0, 5, 10} <= set(r.get("frames", [])))
    obj = {}
    for su in sorted({r["_suite"] for r in keys}):
        kk = [r for r in keys if r["_suite"] == su]
        pres = Counter()
        for r in kk:
            for n, k in r["object_checks"].get("P_name_present", {}).items():
                pres[n] += k
        pf = [v for r in kk for v in r["object_checks"].get("P_full_nonempty", [])]
        sp = [x for r in kk for x in r["object_checks"]["site_px"] if x is not None]
        o = {"gripper_instance_ok": all(r["object_checks"]["gripper_instance"] == "PandaGripper0" for r in kk),
             "site_px_max": max(sp) if sp else None,
             "P_full_nonempty_frac": float(np.mean(pf)) if pf else None,
             "P_names_never_seen": sorted(n for n, k in pres.items() if k == 0)}
        o["status"] = "PASS" if (o["gripper_instance_ok"] and o["site_px_max"] == 0 and (o["P_full_nonempty_frac"] or 0) >= 0.95
                                 and not o["P_names_never_seen"]) else "FAIL"
        obj[su] = o
    return {"n_keys": len(keys), "keys": sorted({(r["_suite"], r["task"], r["ep"]) for r in keys}),
            "frames_missing_0_5_10": frames_bad,
            "cells": {c: {"bad_keys": sorted(v["bad_keys"]), "tolerated_frames": v["tolerated"]} for c, v in cells.items()},
            "object_checks": obj, "gpus": sorted({r["gpu"] for r in keys})}


def read_manip_summary(path, lag):
    s = json.load(open(path))
    k = s["keys"]
    psv = s["per_suite_view"]
    need = [f"{su}/{v}" for su in ("spatial", "object") for v in ("agent", "wrist")]
    why = []
    miss = [x for x in need if x not in psv]
    if miss:
        why.append(f"suite/view missing {miss}")
    if k["visibility_hard_ok"] is not True:
        why.append("visibility_hard_ok not True")
    dec = k["rule_lag"]["decision"]
    if dec != lag:
        why.append(f"lag rule decision {dec} != --lag {lag}")
    gp = sorted({card_type(g) for g in k["gpus"]})
    if gp != ["h200"]:
        why.append(f"manip gpus {gp}")
    return {"sha256": sha256f(path), "rule_lag": k["rule_lag"], "visibility_hard_ok": k["visibility_hard_ok"],
            "status": "FAIL" if why else "PASS", "reasons": why}


# ------------------------------------------------------------------ logs
def server_log(logs_root, block, job, cache):
    key = (block, job)
    if key in cache:
        return cache[key]
    p = os.path.join(logs_root, "server", f"server_{block}_{job}.log")
    r = {"path": p, "cfg": None, "why": [], "refusals": 0}
    if not os.path.isfile(p):
        r["why"].append("server log missing")
    else:
        txt = open(p, errors="replace").read()
        cf = json_lines(txt, "MI_SERVER_CFG")
        if len(cf) != 1 or cf[0][0] is None:
            r["why"].append(f"MI_SERVER_CFG lines {[m for _, m in cf]}")
        else:
            r["cfg"], r["parse"] = cf[0]
        r["refusals"] = len(re.findall(r"request without noise_seed", txt))
        if r["refusals"]:
            r["why"].append(f"noise_seed refusals {r['refusals']}")
    sp = os.path.join(logs_root, "server", f"state_{block}_{job}.jsonl")
    r["state_path"] = sp
    r["state_rows"] = None
    if os.path.isfile(sp):
        rows = defaultdict(list)
        for x in jsonl([sp]):
            rows[int(x["noise_seed"])].append(x["state"])
        r["state_rows"] = rows
    cache[key] = r
    return r


def client_logs(logs_root, block, cell, jobs, tag):
    """c_<block>_<cell>_e<off>[_<tag>]_<job>.log (tools/mi_pi_run.sbatch); only the given tag (default: untagged)."""
    out = []
    pat = re.compile(rf"^c_{re.escape(block)}_{cell}_e\d+{('_' + re.escape(tag)) if tag else ''}_(\d+(?:_p\d+)?)\.log$")
    for f in sorted(glob.glob(os.path.join(logs_root, "clients", f"c_{block}_{cell}_e*.log"))):
        m = pat.match(os.path.basename(f))
        if m and m.group(1) in jobs:
            out.append(f)
    return out


EP_LINE = re.compile(r"^EP task=\d+ ep=\d+ init=\d+ error=.*$")


def check_client_log(path, block, cell, model, suite, lag, srv_cfgs):
    why = []
    txt = open(path, errors="replace").read()
    cf = json_lines(txt, "MI_CFG")
    if len(cf) != 1 or cf[0][0] is None:
        why.append(f"MI_CFG lines {[m for _, m in cf]}")
    else:
        c = cf[0][0]
        want = {"model": MODELS[model]["config"], "suite": f"libero_{suite}", "replan": QUERY_EVERY, "block": block,
                "cell": cell, "query_every": str(QUERY_EVERY), "lag": str(lag), "width": OUTLINE_W,
                "protect": PROTECT, "prep": PREP}
        for k, v in want.items():
            if k not in c:
                why.append(f"MI_CFG missing {k}")
            elif c[k] != v:
                why.append(f"MI_CFG {k}={c[k]!r} != {v!r}")
        if "server" not in c:
            why.append("MI_CFG missing server")
        elif not any(c["server"] == s for s in srv_cfgs if s is not None):
            why.append("MI_CFG server != MI_SERVER_CFG of its job")
    gl = json_lines(txt, "MI_GL_RENDERER")
    if not gl:
        why.append("no MI_GL_RENDERER line")
    for g, m in gl:
        if g is None:
            why.append("MI_GL_RENDERER unparsed")
            continue
        for v in ("physical", "reference"):
            if v not in g or not isinstance(g[v], str) or "H200" not in g[v] or "llvmpipe" in g[v].lower():
                why.append(f"MI_GL_RENDERER {v}={g.get(v)!r}")
    for l in txt.splitlines():  # item 9: no outcome / length in the log
        if l.startswith("EP ") and not EP_LINE.match(l):
            why.append("EP line not blind")
            break
        if l.startswith("FINAL") and not re.match(r"^FINAL episodes \d+$", l.strip()):
            why.append("FINAL line not blind")
            break
        if l.startswith(("ASSERT", "DUMP_ERROR", "LAG_ASSERT")) or "Traceback" in l:
            why.append("assert / traceback in log")
            break
    return why


def check_blind_jsonl(files):
    """item 9: every record blind; returns (n_records, list of reasons). Values are never printed."""
    why, n = [], 0
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            n += 1
            bad = [k for k in REC_FORBIDDEN if k in r]
            if isinstance(r.get("contacts"), dict) and "n_steps" in r["contacts"]:
                bad.append("contacts.n_steps")
            if isinstance(r.get("twin"), dict) and "n_paint" in r["twin"]:
                bad.append("twin.n_paint")
            if r.get("blind") is not True:
                bad.append("blind!=True")
            if bad:
                why.append(f"{os.path.basename(f)}: non-blind record ({sorted(set(bad))})")
                break
    return n, why


# ------------------------------------------------------------------ formal runs
def load_detector():
    p = os.path.expanduser("$PROJ_ROOT/vla_zoo/tools/physics_breakdown_scan.py")
    spec = importlib.util.spec_from_file_location("pbscan", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, sha256f(p)


def oft_scene(oft_root, suite, task, ep, cache):
    """Step-0 qpos / joint names / reference fixtures of the OFT C0 dump of the same key (success refused)."""
    k = (suite, task, ep)
    if k not in cache:
        fs = sorted(glob.glob(os.path.join(oft_root, f"oft_{suite}", "C0", f"t{task}_e{ep}_i*.npz")))
        if len(fs) != 1:
            cache[k] = None
        else:
            z = NoSuccessNpz(fs[0])
            cache[k] = ([str(n) for n in z["joint_names"]], np.asarray(z["qpos"][0]), np.asarray(z["ref_body_pos"]))
            z.close()
    return cache[k]


def check_key(f, block, cell, model, suite, lag, ctx):
    """All per-key checks of one npz. Returns (key, {item: [reasons]}, extracted values for cross checks)."""
    W = defaultdict(list)
    X = {}
    m = re.match(r"t(\d+)_e(\d+)_i(\d+)\.npz$", os.path.basename(f))
    key = (int(m.group(1)), int(m.group(2)))
    z = NoSuccessNpz(f)

    def need(item, *names):
        miss = [n for n in names if n not in z]
        for n in miss:
            W[item].append(f"missing {n}")
        return not miss

    # item 10 labels
    if need("10", "mi_cell", "mr_cell", "mi_block", "task", "ep"):
        if not (str(scalar(z, "mi_cell")) == str(scalar(z, "mr_cell")) == cell):
            W["10"].append("mi_cell / mr_cell != path cell")
        if str(scalar(z, "mi_block")) != block:
            W["10"].append("mi_block != path block")
        if (int(scalar(z, "task")), int(scalar(z, "ep"))) != key:
            W["10"].append("task / ep != file name")
    # items 1, 2
    if need("1", "gpu", "server_gpu"):
        X["gpus"] = [str(scalar(z, "gpu")), str(scalar(z, "server_gpu"))]
        for g in X["gpus"]:
            if "H200" not in g:
                W["1"].append(f"card not H200 ({g})")
    if need("1", "mi_gl_renderer"):
        glr = json.loads(str(scalar(z, "mi_gl_renderer")))
        if not isinstance(glr, dict):
            W["1"].append("mi_gl_renderer not an object")
        else:
            for v in ("physical", "reference"):
                if v not in glr or not isinstance(glr[v], str) or "H200" not in glr[v] or "llvmpipe" in glr[v].lower():
                    W["1"].append(f"mi_gl_renderer {v}={glr.get(v)!r}")
    # frame / query bookkeeping
    steps = qsteps = None
    if need("7", "mr_steps", "query_step"):
        steps = [int(x) for x in z["mr_steps"]]
        qsteps = [int(x) for x in z["query_step"]]
        if not steps or steps[0] != 0 or any(b - a != QUERY_EVERY for a, b in zip(steps, steps[1:])):
            W["7"].append("mr_steps not 0, 5, 10, ...")
        if qsteps != steps[:len(qsteps)]:
            W["7"].append("query_step not the first frames of mr_steps")
    qset = set(qsteps or [])
    # item 4 (client part) + 2.2
    if need("4", "mr_checks", "mi_rows") and steps is not None:
        chk, rows = json.loads(str(scalar(z, "mr_checks"))), json.loads(str(scalar(z, "mi_rows")))
        if len(chk) != len(steps) or len(rows) != len(steps):
            W["4"].append("mr_checks / mi_rows / mr_steps row counts differ")
        else:
            for t, row, mrow in zip(steps, chk, rows):
                for v in ("agent", "wrist"):
                    x = row.get(v)
                    if not isinstance(x, dict) or x.get("same_substep") is not True:
                        W["4"].append("same_substep")
                    if not isinstance(x, dict) or x.get("final_eq_edited") is not True:
                        W["4"].append("final_eq_edited")
                    y = mrow.get(v)
                    if not isinstance(y, dict) or "out_E_diff_px" not in y or y["out_E_diff_px"] != 0:
                        W["4"].append("out_E_diff_px != 0 / missing")
                if t in qset and row.get("model_input_is_edited") is not True:
                    W["4"].append("model_input_is_edited")
            if cell == "C5":
                X["mi_rows_lag"] = [[(r.get(v) or {}).get("lag_step") for v in ("agent", "wrist")] for r in rows]
    if need("S", "mi_S_checks", "mi_S_sent_ok") and steps is not None:
        sc = json.loads(str(scalar(z, "mi_S_checks")))
        ok = [bool(x) for x in z["mi_S_sent_ok"]]
        if len(sc) != len(steps):
            W["S"].append("mi_S_checks row count != frames")
        else:
            for t, s in zip(steps, sc):
                if not isinstance(s, dict) or s.get("pre_eq_post") is not True or s.get("obs_eq_pre") is not True:
                    W["S"].append("pre_eq_post / obs_eq_pre")
                if t in qset and (not isinstance(s, dict) or s.get("sent_eq_pre") is not True):
                    W["S"].append("sent_eq_pre")
        if len(ok) != len(qset) or not all(ok):
            W["S"].append("mi_S_sent_ok")
    # item 7 config
    mc = MODELS[model]
    if need("7", "align_json", "mi_replan", "mi_query_every", "mi_chunk_len", "mi_error", "mi_model", "mi_suite",
            "mi_model_prep", "mi_lag_steps", "mi_server_cfg", "server_host", "server_gpu"):
        al = json.loads(str(scalar(z, "align_json")))
        if not isinstance(al, dict) or al.get("copy_grip_cmd") != 1:
            W["7"].append("copy_grip_cmd")
        if int(scalar(z, "mi_replan")) != QUERY_EVERY or int(scalar(z, "mi_query_every")) != QUERY_EVERY:
            W["7"].append("replan / query_every != 5")
        if int(scalar(z, "mi_chunk_len")) != mc["chunk"]:
            W["7"].append(f"chunk_len != {mc['chunk']}")
        if json.loads(str(scalar(z, "mi_error"))) is not None:
            W["7"].append("mi_error not null")
        if str(scalar(z, "mi_model")) != mc["config"] or str(scalar(z, "mi_suite")) != f"libero_{suite}":
            W["7"].append("mi_model / mi_suite")
        if str(scalar(z, "mi_model_prep")) != PREP:
            W["7"].append("mi_model_prep")
        if int(scalar(z, "mi_lag_steps")) != lag:
            W["7"].append(f"mi_lag_steps != {lag}")
        cfg = json.loads(str(scalar(z, "mi_server_cfg")))
        X["server_cfg"] = cfg
        want = {"config": mc["config"], "pi05": mc["pi05"], "action_horizon": mc["chunk"], "discrete_state_input": False,
                "debug_state_transform": False, "host": str(scalar(z, "server_host")), "gpu": str(scalar(z, "server_gpu"))}
        for k_, v_ in want.items():
            if not isinstance(cfg, dict) or k_ not in cfg:
                W["7"].append(f"server cfg missing {k_}")
            elif cfg[k_] != v_:
                W["7"].append(f"server cfg {k_} != {v_!r}")
        sl = cfg.get("state_log") if isinstance(cfg, dict) else None
        mj = re.search(r"state_(\w+?)_(\d+(?:_p\d+)?)\.jsonl$", str(sl)) if sl else None
        if not mj or mj.group(1) != block:
            W["7"].append("server cfg state_log does not name a job of this block")
        else:
            X["job"] = mj.group(2)
            srv = server_log(ctx["logs_root"], block, X["job"], ctx["srv_cache"])
            if srv["cfg"] is None:
                W["7"].append("job server log: " + "; ".join(srv["why"]))
            elif srv["cfg"] != cfg:
                W["7"].append("npz mi_server_cfg != MI_SERVER_CFG of its job")
    # scene (item 7): step-0 qpos and reference fixtures
    if need("7", "joint_names", "qpos", "ref_body_pos"):
        names = [str(n) for n in z["joint_names"]]
        q = np.asarray(z["qpos"])
        X["scene"] = (names, np.asarray(q[0]), np.asarray(z["ref_body_pos"]))
        if ctx["oft_root"] is not None:
            ref = oft_scene(ctx["oft_root"], suite, key[0], key[1], ctx["oft_cache"])
            if ref is None:
                W["7"].append("no unique OFT C0 dump for the key (scene)")
            elif ref[0] != names or ref[1].shape != X["scene"][1].shape or np.abs(ref[1] - X["scene"][1]).max() > 0 \
                    or ref[2].shape != X["scene"][2].shape or np.abs(ref[2] - X["scene"][2]).max() > 0:
                W["7"].append("scene != OFT C0")
        # detector D (item 6, reported only)
        try:
            ai = [names.index(f"robot0_joint{i}") for i in range(1, 8)]
            A, B, _ = ctx["det"].detect(q, ai, None, -0.5, 1.2)
            X["D"] = bool(A or B)
        except ValueError:
            W["7"].append("arm joints not found (detector D)")
    # item 8
    if need("8", "mi_spec"):
        sp = json.loads(str(scalar(z, "mi_spec")))
        for fn, k_ in (("maskinfo.py", "maskinfo_sha256"), ("maskrepl.py", "maskrepl_sha256")):
            if not isinstance(sp, dict) or sp.get(k_) != FROZEN_SHA[fn]:
                W["8"].append(f"mi_spec {k_} != frozen")
        X["spec_info"] = {k_: (sp.get(k_) if isinstance(sp, dict) else None)
                          for k_ in ("protocol_sha256", "maskinfo_pi_sha256", "cell", "lag_steps", "query_every")}
    # item 9
    if need("9", "mi_blind"):
        if bool(scalar(z, "mi_blind")) is not True:
            W["9"].append("mi_blind not True")
    # item 11 (C5)
    if cell == "C5" and need("11", "mi_lag_step_actual", "mi_lag_step_target", "mi_lag_ok") and steps is not None:
        act = [int(x) for x in z["mi_lag_step_actual"]]
        tgt = [int(x) for x in z["mi_lag_step_target"]]
        if len(act) != len(steps) or len(tgt) != len(steps):
            W["11"].append("lag arrays length != frames")
        else:
            t0 = steps[0]
            if t0 != 0:
                W["11"].append("t0 (mr_steps[0]) != 0")
            if act != tgt:
                W["11"].append("actual != target")
            late = [(t, a) for t, a in zip(steps, act) if t >= t0 + lag]
            if late and sum(a == t - lag for t, a in late) != len(late):
                W["11"].append("frames t >= t0 + lag not all at t - lag")
            if any(a != t0 for t, a in zip(steps, act) if t < t0 + lag):
                W["11"].append("early frames do not hold frame 0")
            if any((tgt_ != (t - lag if t >= t0 + lag else t0)) for t, tgt_ in zip(steps, tgt)):
                W["11"].append("target != rule")
            if "mi_rows_lag" in X and any(any(v != a for v in vv) for vv, a in zip(X["mi_rows_lag"], act)):
                W["11"].append("mi_rows lag_step != actual")
        if not all(bool(x) for x in z["mi_lag_ok"]):
            W["11"].append("mi_lag_ok")
    # item 12
    if need("12", "noise_seed", "mi_noise_master", "mi_model_tag", "mi_noise_md5") and steps is not None:
        ns = np.asarray(z["noise_seed"])
        if ns.dtype.kind not in "iu" or ns.ndim != 1:
            W["12"].append("noise_seed not an integer vector")
        else:
            seeds = [int(x) for x in ns]
            if len(seeds) != len(qset) or len(z["mi_noise_md5"]) != len(seeds):
                W["12"].append("noise_seed / mi_noise_md5 length != queries")
            if seeds != [expected_seed(model, suite, key[0], key[1], q) for q in range(len(seeds))]:
                W["12"].append("noise_seed != sha256 rule")
            X["seed0"] = seeds[0] if seeds else None
            X["noise_md5"] = [str(x) for x in z["mi_noise_md5"]]
        if int(scalar(z, "mi_noise_master")) != MASTER:
            W["12"].append("mi_noise_master != 5478")
        if str(scalar(z, "mi_model_tag")) != model:
            W["12"].append("mi_model_tag != block model")
    # item 13 data (compared across cells later)
    if cell in ("C1", "C8"):
        if need("13", "mi_mask_md5_agent", "mi_mask_md5_wrist"):
            a, w = z["mi_mask_md5_agent"], z["mi_mask_md5_wrist"]
            X["mask0"] = (str(a[0]), str(w[0])) if len(a) and len(w) and str(a[0]) and str(w[0]) else None
            if X["mask0"] is None:
                W["13"].append("frame-0 mask md5 empty")
    # item 14
    if need("14", "mi_S_server_received", "mi_S_server_eq_sent"):
        rec = np.asarray(z["mi_S_server_received"], np.float64)
        if bool(scalar(z, "mi_S_server_eq_sent")) is not True:
            W["14"].append("mi_S_server_eq_sent not True")
        if "job" not in X or X.get("seed0") is None:
            W["14"].append("cannot link to the job's server state log (needs items 7 / 12)")
        else:
            srv = server_log(ctx["logs_root"], block, X["job"], ctx["srv_cache"])
            if srv["state_rows"] is None:
                W["14"].append("server state jsonl missing")
            else:
                cand = srv["state_rows"].get(X["seed0"], [])
                if not cand:
                    W["14"].append("no server state row for the first request's seed")
                elif not any(s is not None and np.asarray(s, np.float64).shape == rec.shape
                             and np.array_equal(np.asarray(s, np.float64), rec) for s in cand):
                    W["14"].append("server state row != mi_S_server_received")
    z.close()
    return key, {k: sorted(set(v)) for k, v in W.items() if v}, X


def scan_block(root, block, lag, ctx, expect):
    model, suite = block.split("_")
    res = {"cells": {}, "physics_breakdown": {}, "gpus": Counter(), "jobs": set(), "spec_info": Counter(),
           "noise_md5_mismatch_keys": []}
    per = {}  # (cell, key) -> X
    for c in CELLS:
        d = os.path.join(root, block, c)
        keys, kres, reasons, pb = set(), {}, Counter(), []
        cell_why = []
        for f in sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz"))):
            try:
                k, W, X = check_key(f, block, c, model, suite, lag, ctx)
            except PermissionError:
                raise
            except Exception as e:  # unreadable / malformed file: the key fails, the gate goes on
                m = re.match(r"t(\d+)_e(\d+)_", os.path.basename(f))
                k, W, X = (int(m.group(1)), int(m.group(2))), {"read": [f"{type(e).__name__}: {e}"]}, {}
            if k in keys:
                kres[k]["items"].setdefault("dup", []).append("duplicate key")
                continue
            keys.add(k)
            per[(c, k)] = X
            for g in X.get("gpus", []):
                res["gpus"][card_type(g)] += 1
            if "job" in X:
                res["jobs"].add(X["job"])
            if X.get("D"):
                pb.append(list(k))
            if "spec_info" in X:
                res["spec_info"][json.dumps({kk: X["spec_info"][kk] for kk in ("protocol_sha256", "maskinfo_pi_sha256")})] += 1
            kres[k] = {"items": W}
        # cell-level: blind jsonl, client logs
        jl = sorted(glob.glob(os.path.join(d, "blind_*.jsonl")))
        jobs = {re.search(r"_(\d+(?:_p\d+)?)\.jsonl$", x).group(1) for x in jl if re.search(r"_(\d+(?:_p\d+)?)\.jsonl$", x)}
        nrec, bw = check_blind_jsonl(jl)
        if not jl:
            cell_why.append("9: no blind_*.jsonl")
        cell_why += ["9: " + w for w in bw]
        res["blind_violation"] = res.get("blind_violation", False) or bool(bw)
        logs = client_logs(ctx["logs_root"], block, c, jobs, ctx["log_tag"])
        if not logs:
            cell_why.append("7: no client log for the jobs of this cell")
        srv_cfgs = [server_log(ctx["logs_root"], block, j, ctx["srv_cache"])["cfg"] for j in sorted(jobs)]
        for L in logs:
            cell_why += [f"7/1/9 {os.path.basename(L)}: {w}" for w in check_client_log(L, block, c, model, suite, lag, srv_cfgs)]
        res["jobs"] |= jobs
        missing = sorted(expect - keys)
        extra = sorted(keys - expect)
        if extra:
            cell_why.append(f"keys outside the expected set: {len(extra)}")
        res["cells"][c] = {"n": len(keys), "missing": len(missing), "missing_keys": [list(k) for k in missing],
                           "cell_fail_reasons": cell_why, "keys": kres, "n_jsonl_records_checked": nrec}
        res["physics_breakdown"][c] = sorted(pb)
    # cross-cell: item 7 scene vs block C0, item 13 C8 vs C1, item 12 noise md5 identical across cells (reported)
    for (c, k), X in per.items():
        W = res["cells"][c]["keys"][k]["items"]
        if c != "C0" and "scene" in X:
            b0 = per.get(("C0", k), {}).get("scene")
            if b0 is None:
                W.setdefault("7", []).append("no C0 scene for the key")
            elif b0[0] != X["scene"][0] or b0[1].shape != X["scene"][1].shape or np.abs(b0[1] - X["scene"][1]).max() > 0 \
                    or b0[2].shape != X["scene"][2].shape or np.abs(b0[2] - X["scene"][2]).max() > 0:
                W.setdefault("7", []).append("scene != block C0")
        if c == "C8":
            a, b = X.get("mask0"), per.get(("C1", k), {}).get("mask0")
            if a is None or b is None:
                W.setdefault("13", []).append("C8 or C1 frame-0 mask md5 unavailable")
            elif a != b:
                W.setdefault("13", []).append("C8 frame-0 mask md5 != C1")
    by_key = defaultdict(list)
    for (c, k), X in per.items():
        if "noise_md5" in X:
            by_key[k].append(X["noise_md5"])
    for k, lists in by_key.items():
        n = min(len(x) for x in lists)
        if any(x[:n] != lists[0][:n] for x in lists):
            res["noise_md5_mismatch_keys"].append(list(k))
    # statuses
    for c in CELLS:
        cr = res["cells"][c]
        bad = {}
        rs = Counter()
        for k, v in cr["keys"].items():
            if v["items"]:
                bad[k] = v["items"]
                for it in v["items"]:
                    rs[it] += 1
            v["status"] = "FAIL" if v["items"] else "PASS"
        cr["bad_keys"] = sorted([list(k) for k in bad])
        cr["reasons_by_item"] = dict(rs)
        cr["status"] = "FAIL" if (cr["missing"] or cr["cell_fail_reasons"]) else "PASS"
        cr["keys"] = {f"t{k[0]}_e{k[1]}": v for k, v in sorted(cr["keys"].items())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--manip", nargs="*", default=[])
    ap.add_argument("--t0", nargs="*", default=[])
    ap.add_argument("--manip-summary", default="")
    ap.add_argument("--logs-root", default="logs")
    ap.add_argument("--embody-dir", default="embody")
    ap.add_argument("--log-tag", default="", help="client-log tag of a tagged smoke root (e.g. rerun); formal: none")
    ap.add_argument("--oft-ref", default=os.path.expanduser("$PROJ_ROOT/maskinfo/out/runs"),
                    help="root of the OFT maskinfo dumps for the cross-model scene check; 'none' = not checked")
    ap.add_argument("--blocks", default=",".join(BLOCKS))
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--eps", default="20-29")
    ap.add_argument("--lag", type=int, default=15)
    a = ap.parse_args()
    expect = {(t, e) for t in parse_range(a.tasks) for e in parse_range(a.eps)}
    out = {"protocol_sha256": sha256f(a.protocol), "script_sha256": sha256f(os.path.abspath(__file__)),
           "argv": sys.argv[1:], "expected_keys_per_cell": len(expect), "lag": a.lag, "not_checked": NOT_CHECKED}
    out["manip"] = summarize_manip(a.manip) if a.manip else None
    out["t0"] = summarize_t0(a.t0) if a.t0 else None
    out["manip_summary"] = read_manip_summary(a.manip_summary, a.lag) if a.manip_summary else None
    out["embody_sha256"] = {fn: (sha256f(os.path.join(a.embody_dir, fn)) if os.path.isfile(os.path.join(a.embody_dir, fn)) else None)
                            for fn in FROZEN_SHA}
    det, dsha = load_detector()
    out["detector_sha256"] = dsha
    ctx = {"logs_root": a.logs_root, "log_tag": a.log_tag, "srv_cache": {}, "oft_cache": {}, "det": det,
           "oft_root": None if a.oft_ref == "none" else a.oft_ref}
    out["blocks"] = {}
    for b in a.blocks.split(","):
        if not os.path.isdir(os.path.join(a.runs, b)):
            out["blocks"][b] = {"present": False, "block_ok": False, "block_fail_reasons": ["block directory missing"]}
            continue
        model, suite = b.split("_")
        r = scan_block(a.runs, b, a.lag, ctx, expect)
        items = {}
        types = set(r["gpus"])
        ct = next(iter(types)) if len(types) == 1 else ("mixed" if types else None)
        # 1
        f1 = []
        rd = (out["manip"] or {}).get("render_det", {})
        if out["manip"] is None:
            f1.append("--manip missing (render determinism)")
        elif rd.get("h200", {}).get("status") != "PASS":
            f1.append(f"render determinism on h200 not PASS ({rd.get('h200')})")
        items["1"] = f1
        # 2
        items["2"] = [] if ct in ALLOWED else [f"card type {ct} ({dict(r['gpus'])})"]
        # 3
        f3 = []
        if out["t0"] is None:
            f3.append("--t0 missing")
        else:
            t0keys = {(t, e) for s, t, e in out["t0"]["keys"] if s == suite}
            t0bad = {c: {(t, e) for s, t, e in out["t0"]["cells"][c]["bad_keys"] if s == suite} for c in CELLS}
            fbad = {(t, e) for s, t, e in out["t0"]["frames_missing_0_5_10"] if s == suite}
            if suite == "object" and out["t0"]["object_checks"].get("object", {}).get("status") != "PASS":
                f3.append(f"object checks {out['t0']['object_checks'].get('object')}")
            for c in CELLS:
                for kn, v in r["cells"][c]["keys"].items():
                    t, e = (int(x) for x in re.match(r"t(\d+)_e(\d+)", kn).groups())
                    why = []
                    if (t, e) not in t0keys:
                        why.append("key not in t0 check")
                    if (t, e) in t0bad[c]:
                        why.append("t0 check failed")
                    if (t, e) in fbad:
                        why.append("t0 frames 0/5/10 missing")
                    if why:
                        v["items"]["3"] = why
                        v["status"] = "FAIL"
                        r["cells"][c]["reasons_by_item"]["3"] = r["cells"][c]["reasons_by_item"].get("3", 0) + 1
                        if [t, e] not in r["cells"][c]["bad_keys"]:
                            r["cells"][c]["bad_keys"] = sorted(r["cells"][c]["bad_keys"] + [[t, e]])
        items["3"] = f3
        # 5
        ph = ((out["manip"] or {}).get("physics") or {}).get(suite)
        items["5"] = [] if ph and ph["status"] == "PASS" else [f"physics invariance ({ph})"]
        # M: the remaining snapshot-mode manipulation checks (protocol 2.6; frozen gate "manipulation checks not PASS")
        items["M"] = ["--manip missing"] if out["manip"] is None else (
            [f"manip violations {out['manip']['violations']}"] if out["manip"]["violations"] or not out["manip"]["n_keys"] else [])
        # 7 (block): server logs of every job
        f7 = []
        for j in sorted(r["jobs"]):
            s = server_log(a.logs_root, b, j, ctx["srv_cache"])
            f7 += [f"job {j}: {w}" for w in s["why"] if "refusals" not in w]
        if not r["jobs"]:
            f7.append("no job linked")
        items["7"] = f7
        # 8 (block): on-disk frozen files
        items["8"] = [f"{fn} sha256 {v} != frozen" for fn, v in out["embody_sha256"].items() if v != FROZEN_SHA[fn]]
        # 9 (block): any non-blind record rejects the block
        items["9"] = ["non-blind record in a blind_*.jsonl"] if r.pop("blind_violation", False) else []
        # 12 (block): refusals
        items["12"] = [f"job {j}: noise_seed refusals" for j in sorted(r["jobs"])
                       if server_log(a.logs_root, b, j, ctx["srv_cache"])["refusals"]]
        # 15
        ms = out["manip_summary"]
        items["15"] = ["--manip-summary missing"] if ms is None else ms["reasons"]
        # per-item roll-up over keys and cells
        roll = {}
        for it in ITEMS:
            nk = sum(r["cells"][c]["reasons_by_item"].get(it, 0) for c in CELLS)
            bl = items.get(it, [])
            roll[it] = {"status": "FAIL" if (bl or nk) else "PASS", "block_reasons": bl, "bad_key_count": nk}
        roll["6"] = {"status": "REPORTED", "keys_flagged": {c: len(v) for c, v in r["physics_breakdown"].items()},
                     "bad_key_count": sum(r["cells"][c]["reasons_by_item"].get("6", 0) for c in CELLS)}
        fails = [f"item {it}: {w}" for it in ITEMS for w in items.get(it, [])]
        r["items"] = roll
        r["card_type"] = ct
        r["block_fail_reasons"] = fails
        r["cells_status"] = {c: r["cells"][c]["status"] for c in CELLS}
        r["block_ok"] = not fails and all(x["status"] == "PASS" for x in r["cells"].values())
        r["gpus"] = dict(r["gpus"])
        r["jobs"] = sorted(r["jobs"])
        r["spec_info_reported"] = dict(r["spec_info"])
        del r["spec_info"]
        r["present"] = True
        out["blocks"][b] = r
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: list(o) if isinstance(o, (set, tuple)) else str(o))
    m = out["manip"] or {}
    print(f"manip: {m.get('status')} keys {m.get('n_keys')} physics {m.get('physics')} render_det {json.dumps(m.get('render_det'))}")
    if out["t0"]:
        print(f"t0: keys {out['t0']['n_keys']} " + " ".join(f"{c}:{len(v['bad_keys'])}" for c, v in out['t0']['cells'].items()))
    print(f"manip_summary: {(out['manip_summary'] or {}).get('status')}  embody sha ok: "
          f"{all(out['embody_sha256'][k] == v for k, v in FROZEN_SHA.items())}")
    for b, r in out["blocks"].items():
        if not r.get("present"):
            print(f"{b}: absent")
            continue
        print(f"{b}: card {r['card_type']} block_ok {r['block_ok']} jobs {r['jobs']}")
        print("  items: " + " ".join(f"{it}:{v['status']}" + (f"({v['bad_key_count']}k)" if v.get('bad_key_count') else "")
                                    for it, v in r["items"].items()))
        print("  cells: " + " ".join(f"{c}:{x['status']}/{x['n']}/miss{x['missing']}/bad{len(x['bad_keys'])}"
                                    for c, x in r["cells"].items()))
        for w in r["block_fail_reasons"]:
            print("  block fail:", w)


if __name__ == "__main__":
    main()
