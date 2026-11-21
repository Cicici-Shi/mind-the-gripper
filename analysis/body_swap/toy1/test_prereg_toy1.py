"""Synthetic-data tests for prereg_toy1.py (no real success data is read; every dump here is generated).

Usage (cluster, CPU only):
  source ~/miniforge3/etc/profile.d/conda.sh; conda activate oft
  srun -p normal -c 4 --mem 8G -t 20 python -B $PROJ_ROOT/embody_toy/analysis/test_prereg_toy1.py \
      $PROJ_ROOT/embody_toy/out/prereg_test/runs $PROJ_ROOT/embody_toy/out/prereg_test/protocol_toy1.md \
      $PROJ_ROOT/embody_toy/out/prereg_test/protocol_toy1_astra_decisions.md [--stages-dir $PROJ_ROOT/embody/analysis]
Each scenario writes OUTDIR/<name>/{runs/,gate.json,shift.jsonl,parts.jsonl,results.txt,verdicts.json} and asserts
on the verdict json / output text. Prints PASS/FAIL per scenario; exit code 1 if any fails.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prereg_toy1.py")
OUT, PROTO, DECI = sys.argv[1], sys.argv[2], sys.argv[3]
STAGES_DIR = sys.argv[sys.argv.index("--stages-dir") + 1] if "--stages-dir" in sys.argv else os.path.expanduser("$PROJ_ROOT/embody/analysis")
sys.path.insert(0, STAGES_DIR)
sys.dont_write_bytecode = True
import stages  # noqa: E402

OBJS = stages.task_objects()
PSHA = hashlib.sha256(open(PROTO, "rb").read()).hexdigest()
DSHA = hashlib.sha256(open(DECI, "rb").read()).hexdigest()
DIRS_ALL = {"G_map20": "toy_g_map20"}
DIRS = {"S": "toy_s", "G": "toy_g", "G_T^fc": "toy_g_tfc", "G_hid": "toy_g_hid", "G_T": "toy_g_t", "S_hid": "toy_s_hid", "G_cam": "toy_g_cam"}
KEYS = [(t, e) for t in range(10) for e in range(40, 50)]
REF_NAMES = ["table", "cabinet", "plate_fix", "stove"]
T = 48

# ------------------------------------------------------------------------------------------- closed-loop dumps
def traj(task, kind):
    """qpos / eef / action for stages.py: kind in success, no_reach, no_grasp."""
    target, plate = OBJS[task]
    names = [f"robot0_joint{i}" for i in range(1, 8)] + ["gripper0_finger_joint1", "gripper0_finger_joint2",
                                                          target + "_joint0", plate + "_joint0"]
    q = np.zeros((T, 9 + 14))
    bowl = np.tile([0.0, 0.0, 0.90, 1, 0, 0, 0], (T, 1)).astype(float)
    pl = np.tile([0.25, 0.0, 0.90, 1, 0, 0, 0], (T, 1)).astype(float)
    eef = np.tile([0.0, 0.3, 1.10], (T, 1)).astype(float)
    act = np.zeros((T, 7))
    act[:, 6] = -1
    if kind != "no_reach":
        eef[10:] = [0.0, 0.0, 0.95]
    if kind == "success":
        act[15:35, 6] = 1
        bowl[18:, 2] = 0.96
        bowl[25:, 0] = 0.25
        eef[25:, 0] = 0.25
    q[:, 9:16], q[:, 16:23] = bowl, pl
    return names, q, eef, act


def ref_pos(t, e, bad=False):
    rng = np.random.default_rng(t * 100 + e)
    p = rng.normal(0, 0.3, (len(REF_NAMES), 3))
    if bad:
        p[0, 0] += 1e-6  # 0.001 mm
    return p


def write_dump(path, t, e, succ, cell, over):
    kind = "success" if succ else ("no_reach" if (t + e) % 2 else "no_grasp")
    names, q, eef, act = traj(t, kind)
    fam_g = cell.startswith("G")
    align = {"copy_grip_cmd": 1, "grp_map": "kp100" if fam_g else None,
             "hide_gripper": "all" if cell in ("G_hid", "S_hid") else None,
             "twin_cam": {"G_T^fc": "phys", "G_T": "twin"}.get(cell), "wrist_cam_dz_mm": 6.4 if cell == "G_cam" else 0.0}
    align.update(over.get("align", {}).get((t, e), {}))
    d = dict(success=np.bool_(succ), task=t, ep=e, init=e, action=act, eef_pos=eef, eef_quat=np.tile([1, 0, 0, 0.0], (T, 1)),
             gripper_qpos=np.zeros((T, 2)), qpos=q, joint_names=np.array(names), ref_body_names=np.array(REF_NAMES),
             ref_body_pos=ref_pos(t, e, bad=(t, e) in over.get("badref", ())), ref_body_quat=np.tile([1, 0, 0, 0.0], (len(REF_NAMES), 1)),
             contact_arm_env_contact_steps=np.int64(3 if not succ else 0), contact_first_contact=np.int64(12),
             contact_pairs=json.dumps({"robot0_link7|table": 3} if not succ else {}), contact_stall_steps=np.int64(0),
             contact_first_grip_obj=np.int64(15 if succ else -1),
             host="node17", gpu=over.get("gpu", "NVIDIA A100 80GB PCIe"), align_json=json.dumps(align))
    if cell in ("G_T^fc", "G_T"):
        pe = np.full(T, 0.3)
        nbad = over.get("ik_bad", 0)
        pe[: 8 * nbad: 8] = 2.0
        d.update(twin_pos_err_mm=pe, twin_rot_err_deg=np.full(T, 0.2), twin_stale=np.int64(over.get("stale", 0)),
                 twin_cam_err_mm=np.full(T, 6.4 if cell == "G_T" else over.get("cam_err", 0.0)), twin_cam_err_deg=np.zeros(T))
    np.savez(path, **d)


def make_succ(sr, perm):
    """sr = success count out of 100 (per 100 keys); perm = rank order of keys."""
    return {k: bool(perm[i] < sr) for i, k in enumerate(KEYS)}


def write_runs(root, srs, indep=(), over=None, eps=range(40, 50), drop=None):
    over = over or {}
    base = np.random.default_rng(1).permutation(100)
    for cell, sr in srs.items():
        perm = np.random.default_rng(sum(map(ord, cell)) + 7).permutation(100) if cell in indep else base
        succ = make_succ(sr, perm)
        d = os.path.join(root, DIRS.get(cell) or DIRS_ALL[cell])
        os.makedirs(d, exist_ok=True)
        co = over.get(cell, {})
        for (t, e), s in succ.items():
            if e not in eps or (drop and (cell, t, e) in drop):
                continue
            ce = dict(co)
            if "ik_bad_key" in co:
                ce["ik_bad"] = co["ik_bad_key"].get((t, e), 0)
            write_dump(os.path.join(d, f"t{t}_e{e}_i{e}.npz"), t, e, s, cell, ce)


def write_gate(path, status=None, missing=(), drop_arm=False):
    status = status or {}
    cells = {}
    for c, d in DIRS.items():
        if c in missing:
            continue
        st = status.get(c, "PASS")
        items = {"t0": {"status": st, "value": 0.05}}
        if c == "G_T^fc" and not drop_arm:
            items["arm_thirdview_consistency"] = {"status": "PASS", "value": 0.0}
        cells[d] = {"status": st, "items": items}
    json.dump({"cells": cells}, open(path, "w"))


# ------------------------------------------------------------------------------------------- probe jsonl
def rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def shift_rows(app, rng, frac, K, leak=0.05, kc=0.3, dose=2.0, noise=0.0, bad_centroid=False, eps=range(40, 46), drop=0):
    rows = []
    keys = [(t, e) for t in range(10) for e in eps]
    nneg = int(round(frac * len(keys)))
    signs = np.array([-1] * nneg + [1] * (len(keys) - nneg))
    np.random.default_rng(5).shuffle(signs)
    for (t, e), s in zip(keys, signs):
        k = K * rng.uniform(0.8, 1.2)
        for j, (step, ph) in enumerate([(16, "approach"), (24, "approach"), (32, "pregrasp"), (40, "pregrasp"), (48, "pregrasp")]):
            if drop and j == 0 and t == 0:
                continue
            fr = np.random.default_rng(t * 1000 + e * 10 + j)  # same frame state for both appearances
            R = rand_rot(fr)
            U = rand_rot(fr)
            ux, uy = U[:, 0], U[:, 1]
            a0 = fr.normal(0, 0.3, (8, 7))
            a0[:, 6] = np.where(np.arange(8) < 4, -1.0, 1.0)
            acts = {}
            for c in ["0", "0p", "x-10", "x-5", "x+5", "x+10", "y-5", "y+5", "cam-5", "cam+5"]:
                a = a0.copy()
                a[:, :3] += rng.normal(0, 0.002, (8, 3)) if c != "0" else 0
                w = np.zeros(3)
                if c[0] in "xy":
                    dl = float(c[1:])
                    gain = s * k / 10 if abs(dl) == 5 else s * k / 10 * dose / 2
                    u = ux if c[0] == "x" else uy
                    w = gain * dl * u + (leak * dl / 10 * uy if c[0] == "x" else 0)
                elif c.startswith("cam"):
                    w = kc * rng.choice([-1, 1]) * float(c[3:]) / 10 * ux
                elif c == "0p":
                    w = rng.normal(0, noise, 3) if noise else np.zeros(3)
                    if not noise:
                        a = a0.copy()
                a[:, :3] += (R.T @ (w / 8))[None, :]
                acts[c] = a.tolist()
            cx = 20.0 if bad_centroid else 8.0
            rows.append({"appearance": app, "task": t, "ep": e, "init": e, "step": step, "phase": ph,
                         "u_x": ux.tolist(), "u_y": uy.tolist(), "R_base": R.tolist(), "actions": acts, "raw": acts,
                         "cam_err_mm": 0.0, "cam_err_deg": 0.0,
                         "centroid_px": {"x+5": [cx, 0.5], "x-5": [-cx, -0.5], "x+10": [2 * cx, 1], "x-10": [-2 * cx, -1]},
                         "same_0_0p": {"pixels": not noise, "actions": not noise}})
    return rows


def write_shift(path, panda, robotiq):
    rng = np.random.default_rng(3)
    rows = shift_rows("panda", rng, **panda) + shift_rows("robotiq", rng, **robotiq)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_parts(path, r=None, with56=True):
    r = r or {"c2": 0.2, "c3": 0.4, "c4": 0.8, "c5": 0.2, "c6": 0.8}
    rng = np.random.default_rng(4)
    with open(path, "w") as f:
        for t in range(10):
            for e in range(40, 46):
                for step in (16, 24, 32, 40, 48):
                    c1 = rng.normal(0, 0.3, (8, 7))
                    v = rng.normal(0, 0.1, (8, 7))
                    A = {"c1": c1.tolist(), "c0": (c1 + v).tolist()}
                    for c, x in r.items():
                        if c in ("c5", "c6") and not with56:
                            continue
                        A[c] = (c1 + x * v).tolist()
                    f.write(json.dumps({"task": t, "ep": e, "init": e, "step": step, "phase": "pregrasp", "actions": A}) + "\n")


# ------------------------------------------------------------------------------------------- runner
BASE = {"S": 98, "G": 54, "G_T": 90, "G_T^fc": 88, "G_hid": 85, "S_hid": 97, "G_cam": 52}
PDET = dict(frac=0.9, K=1.0)
PNON = dict(frac=0.5, K=1.0)


def run(name, srs=None, keys="all", indep=(), over=None, eps=range(40, 50), drop=None, gate="default", gate_kw=None,
        shift=None, parts="default", extra=(), expect_rc=0, sha=None, dsha=None, pre_out=False):
    d = os.path.join(OUT, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    runs = os.path.join(d, "runs")
    write_runs(runs, srs if srs is not None else BASE, indep, over, eps, drop)
    args = [sys.executable, SCRIPT, "--expect-protocol-sha", sha or PSHA, "--expect-decisions-sha", dsha or DSHA,
            "--keys", keys, "--out", os.path.join(d, "results.txt"), "--verdict-json", os.path.join(d, "verdicts.json"),
            "--runs", runs, "--protocol", PROTO, "--decisions", DECI, "--stages-dir", STAGES_DIR]
    if gate is not None:
        write_gate(os.path.join(d, "gate.json"), **(gate_kw or {}))
        args += ["--gate", os.path.join(d, "gate.json")]
    if shift is not None:
        write_shift(os.path.join(d, "shift.jsonl"), *shift)
        args += ["--probe-shift", os.path.join(d, "shift.jsonl")]
    if parts is not None:
        write_parts(os.path.join(d, "parts.jsonl"), **({} if parts == "default" else parts))
        args += ["--probe-parts", os.path.join(d, "parts.jsonl")]
    if pre_out:
        open(os.path.join(d, "results.txt"), "w").write("old\n")
    p = subprocess.run(args + list(extra), capture_output=True, text=True)
    assert p.returncode == expect_rc, f"rc {p.returncode} != {expect_rc}\n{p.stderr[-3000:]}"
    if expect_rc:
        return None, p.stderr
    V = json.load(open(os.path.join(d, "verdicts.json")))
    txt = open(os.path.join(d, "results.txt")).read()
    return V, txt


RESULTS = []


def scenario(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, "PASS", ""))
    except Exception as e:  # noqa
        RESULTS.append((fn.__name__, "FAIL", f"{e!r}\n{traceback.format_exc()[-1500:]}"))


# ---- closed loop: Q0 / Q1
def t01_go_interference():
    V, txt = run("go_interference", shift=(PDET, dict(frac=0.5, K=1.0)))
    assert V["complete"] and V["q0"]["raw"] == "GO" and V["q0"]["state"] == "ok", V["q0"]
    assert V["q1"]["raw"] == "干扰为主" and V["q1"]["state"] == "ok", V["q1"]
    assert "分支 A" in V["branch"] and "路线筛选" in V["branch"], V["branch"]
    assert V["n"]["R_fc"] == 100 and V["n"]["C_R"] == 100
    assert "失败类型" in txt and "no_reach" in txt and "no_grasp" in txt, "stages labels"
    assert "McNemar" in txt and "ep40-44" in txt


def t02_go_cue_rho():
    V, _ = run("go_cue_rho", {**BASE, "G_hid": 58})
    assert V["q0"]["raw"] == "GO" and V["q1"]["raw"] == "线索为主" and V["q1"]["rho"] <= 0.3, V["q1"]
    assert "分支 B" in V["branch"]


def t03_go_cue_hs():
    V, _ = run("go_cue_hs", {**BASE, "G_hid": 70, "S_hid": 70})
    assert V["q1"]["raw"] == "线索为主" and 0.3 < V["q1"]["rho"] < 0.7, V["q1"]


def t04_go_mixed():
    V, txt = run("go_mixed", {**BASE, "G_hid": 70})
    assert V["q1"]["raw"] == "混合/不确定", V["q1"]
    assert "阴影是未控因素" in txt and "人工" in V["branch"]


def t05_camera():
    V, _ = run("camera", {**BASE, "G_T^fc": 56})
    assert V["q0"]["raw"] == "转向相机", V["q0"]
    assert V["q1"]["state"] == "not_entered" and "相机转向" in V["branch"]


def t06_camera_cp_uncertain():
    V, txt = run("camera_cp_unc", {**BASE, "G_T^fc": 56, "G_T": 72}, indep=("G_T",))
    est = V["est"]
    assert est["R_fc"]["ci95"][1] < 10 and est["R_T"]["point"] >= 15 and est["C_P"]["ci95"][0] <= 5, est
    assert V["q0"]["raw"] == "不确定", V["q0"]


def t07_not_detected():
    V, txt = run("q0_not_detected", {**BASE, "G_T^fc": 66}, indep=("G_T^fc",))
    assert V["q0"]["raw"] == "未检出", (V["q0"], V["est"]["R_fc"])
    assert "外观为主" not in txt.split("==== §5.1")[1].split("====")[0], "无信号不得写外观为主"


def t08_rt_low_no_longer_blocks():
    V, txt = run("rt_low", {**BASE, "G_T": 60})
    assert V["est"]["R_T"]["point"] < 15
    assert V["q0"]["raw"] == "GO" and V["q0"]["state"] == "ok", "R_T 不足不再禁止 Q0"
    assert V["q1"]["raw"] == "干扰为主"


# ---- gates
def t09_gate_copy_fail():
    V, txt = run("gate_copy_fail", over={"G_T^fc": {"align": {(3, 41): {"copy_grip_cmd": 0}}}})
    assert V["gates"]["G_T^fc"] == "FAIL" and V["q0"]["state"] == "dnc" and V["q0"]["raw"] == "GO", V["q0"]
    assert "仅供参考" in txt and V["q1"]["state"] == "not_entered"


def t10_gate_grpmap_fail():
    V, _ = run("gate_grpmap_fail", over={"G": {"align": {(0, 40): {"grp_map": "kp20"}}}})
    assert V["gates"]["G"] == "FAIL" and V["q0"]["state"] == "dnc"


def t11_gate_ik_fail():
    V, txt = run("gate_ik_fail", over={"G_T^fc": {"ik_bad_key": {(t, 40): 1 for t in range(10)}}})
    # 10 bad of 100 x 6 query frames = 1.67% >= 1%
    assert V["gates"]["G_T^fc"] == "FAIL" and "IK 残差" in txt, V["gates"]


def t12_gate_ik_pass_below_1pct():
    V, _ = run("gate_ik_pass", over={"G_T^fc": {"ik_bad_key": {(0, 40): 1, (1, 40): 1}}})
    assert V["gates"]["G_T^fc"] == "PASS", V["gates"]


def t13_gate_stale_and_cam():
    V, _ = run("gate_stale", over={"G_T^fc": {"stale": 1}})
    assert V["gates"]["G_T^fc"] == "FAIL"
    V, _ = run("gate_cam", over={"G_T^fc": {"cam_err": 0.02}})
    assert V["gates"]["G_T^fc"] == "FAIL"


def t14_gate_json_fail_q1():
    V, txt = run("gate_json_fail", gate_kw={"status": {"G_hid": "FAIL"}})
    assert V["gates"]["G_hid"] == "FAIL" and V["q0"]["state"] == "ok"
    assert V["q1"]["state"] == "dnc" and V["q1"]["raw"] == "干扰为主", V["q1"]
    assert "Q1 不判定" in V["branch"]


def t15_gate_json_unk():
    V, txt = run("gate_json_unk", gate_kw={"missing": ("G_hid",)})
    assert V["gates"]["G_hid"] == "UNK" and V["q1"]["state"] == "unverified" and V["q1"]["raw"] == "干扰为主", V["q1"]
    assert "门槛未核实" in txt and "分支 A" in V["branch"]


def t16_gate_arm_item_missing():
    V, txt = run("gate_arm_missing", gate_kw={"drop_arm": True})
    assert V["gates"]["G_T^fc"] == "UNK" and V["q0"]["state"] == "unverified", V["q0"]
    assert "arm_thirdview_consistency" in txt


def t17_no_gate_json():
    V, _ = run("no_gate_json", gate=None)
    assert all(v == "UNK" for v in V["gates"].values()), V["gates"]
    assert V["q0"]["state"] == "unverified"


def t18_fixcheck_exclusion():
    V, txt = run("fixcheck", over={"G_hid": {"badref": ((2, 44),)}})
    assert V["n"]["R_fc"] == 99 and V["n"]["H_S"] == 99, V["n"]
    assert "剔除键 t2 e44" in txt and V["gates"]["G_hid"] == "PASS"


# ---- completeness / keys
def t19_first_shard_only():
    V, txt = run("first_only", keys="first", eps=range(40, 45))
    assert V["complete"] and V["n"]["R_fc"] == 50 and V["q0"]["raw"] == "GO", (V["n"], V["q0"])


def t20_first_shard_only_but_all_requested():
    V, txt = run("first_only_all", keys="all", eps=range(40, 45))
    assert not V["complete"] and V["q0"]["state"] == "incomplete"
    assert "§4 估计量" not in txt and "未完成" in txt


def t21_one_missing_episode_blocks_shard():
    drop = {("G_cam", 7, 47)}
    V, txt = run("hole_shard2_all", keys="all", drop=drop)
    assert not V["complete"] and V["shards"] == {"ep40-44": True, "ep45-49": False}, V["shards"]
    V, _ = run("hole_shard2_first", keys="first", drop=drop)
    assert V["complete"] and V["n"]["R_fc"] == 50


def t22_gcam_missing_is_incomplete():
    srs = {k: v for k, v in BASE.items() if k != "G_cam"}
    V, _ = run("no_gcam", srs)
    assert not V["complete"] and V["gates"]["G_cam"] == "MISSING"


def t23_map20_ignored():
    V, txt = run("map20_ignored", {**BASE, "G_map20": 10})
    assert os.path.isdir(os.path.join(OUT, "map20_ignored", "runs", "toy_g_map20"))
    assert "G_map20" not in V["gates"] and "M" not in V["est"] and "toy_g_map20" not in txt


# ---- discipline
def t24_refuse_wrong_sha():
    _, err = run("wrong_sha", sha="0" * 64, expect_rc=2)
    assert "拒绝运行" in err and not os.path.exists(os.path.join(OUT, "wrong_sha", "results.txt"))
    _, err = run("wrong_dsha", dsha="0" * 64, expect_rc=2)
    assert "裁定" in err


def t25_refuse_overwrite():
    _, err = run("overwrite", pre_out=True, expect_rc=3)
    assert "拒绝覆盖" in err and open(os.path.join(OUT, "overwrite", "results.txt")).read() == "old\n"


def t26_sha_printed():
    V, txt = run("sha_printed")
    assert PSHA in txt and DSHA in txt and hashlib.sha256(open(SCRIPT, "rb").read()).hexdigest() in txt


# ---- §5.3 probe
def t27_probe_row2():
    V, txt = run("probe_row2", shift=(PDET, PNON))
    q = V["q2"]
    assert q["panda"] == "检出" and q["robotiq"] == "未检出" and q["reading"] == "失效" and q["row"] == 2, q
    assert "不能证明效应来自指尖" in txt and "未检出不等于不存在" in txt


def t28_probe_row1():
    V, _ = run("probe_row1", shift=(PNON, PDET))
    assert V["q2"]["panda"] == "未检出" and V["q2"]["row"] == 1, V["q2"]


def t29_probe_row3():
    V, _ = run("probe_row3", shift=(PDET, dict(frac=0.9, K=0.8)))
    assert V["q2"]["robotiq"] == "检出" and V["q2"]["reading"] == "读得懂" and V["q2"]["row"] == 3, V["q2"]


def t30_probe_undefined_half():
    V, _ = run("probe_half", shift=(PDET, dict(frac=0.9, K=0.3)))
    assert V["q2"]["robotiq"] == "检出" and V["q2"]["reading"] == "未定义" and V["q2"]["row"] is None, V["q2"]


def t31_probe_dose_necessary():
    V, txt = run("probe_nodose", shift=(dict(frac=0.9, K=1.0, dose=1.0), PNON))
    assert V["q2"]["panda"] == "未检出", V["q2"]  # sign and specificity pass, dose fails


def t32_probe_leak():
    V, _ = run("probe_leak", shift=(dict(frac=0.9, K=1.0, leak=0.8), PNON))
    assert V["q2"]["panda"] == "未检出", V["q2"]


def t33_probe_cam_not_criterion():
    # camera response larger than D_x: old 1.2x rule would have said "absent"; now camera is descriptive only
    V, txt = run("probe_bigcam", shift=(dict(frac=0.9, K=1.0, kc=3.0), PNON))
    assert V["q2"]["panda"] == "检出", V["q2"]
    assert "不是阴性 sham" in txt


def t34_probe_noise_floor_reported():
    V, txt = run("probe_noise", shift=(dict(frac=0.9, K=1.0, noise=0.05), PNON))
    assert V["q2"]["panda"] == "检出" and "0 与 0′ 不同的帧 300/300" in txt, txt[-200:]


def t35_probe_centroid_fail():
    V, txt = run("probe_centroid", shift=(dict(frac=0.9, K=1.0, bad_centroid=True), PNON))
    assert V["q2"]["panda"] is None and V["q2"]["row"] is None and "操纵检查" in txt and "仅供参考" in txt, V["q2"]


def t36_probe_pairing_fail():
    V, txt = run("probe_unpaired", shift=(PDET, dict(frac=0.5, K=1.0, drop=1)))
    assert V["q2"]["panda"] is None and V["q2"]["robotiq"] is None, V["q2"]
    assert "两外观同帧配对" in txt


def t37_probe_source_mixed():
    V, txt = run("probe_mixed_src", shift=(dict(frac=0.9, K=1.0, eps=(10, 11, 12, 40, 41, 42)), dict(frac=0.5, K=1.0, eps=(10, 11, 12, 40, 41, 42))))
    assert V["q2"]["panda"] is None and "混用" in txt, V["q2"]


# ---- §5.4 parts
def t38_parts_readings():
    V, txt = run("parts")
    q = V["q3"]
    assert abs(q["c2"] - 0.2) < 0.02 and abs(q["c6"] - 0.8) < 0.02, q
    assert q["readings"]["r2"] == "与 Q1 干扰一致" and q["readings"]["r3_r4"].startswith("c3") and q["readings"]["c5_c6"] == "手指承担效应"


def t39_parts_mix_and_missing56():
    V, _ = run("parts_mix", parts={"r": {"c2": 0.5, "c3": 0.6, "c4": 0.5, "c5": 0.5, "c6": 0.6}})
    assert V["q3"]["readings"]["c5_c6"].startswith("部件不可分") and V["q3"]["readings"]["r3_r4"].startswith("c4")
    V, txt = run("parts_no56", parts={"with56": False})
    assert V["q3"]["c5"] is None and "门槛未核实" in txt and V["q3"]["readings"]["c5_c6"] is None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t") and k[1:3].isdigit() and callable(v)]
    for fn in tests:
        scenario(fn)
        print(f"{RESULTS[-1][1]}  {RESULTS[-1][0]}", flush=True)
        if RESULTS[-1][1] == "FAIL":
            print(RESULTS[-1][2], flush=True)
    nf = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS) - nf}/{len(RESULTS)} scenarios passed")
    sys.exit(1 if nf else 0)
