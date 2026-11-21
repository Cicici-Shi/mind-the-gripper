"""rescue pilot smoke check (blind). For rs_smoke_{oftc,ofti}_sp x 3 cells: exactly keys t0 e0,e1; pf_cell_env equals
cells_next.sh switches (+EMBODY_DIR, M1_FASTSYNC); body really Robotiq/Panda (finger joint present, 6 armature dofs);
twin fields present + stale 0 + phys-cam pinned only in GTfc2; pf_blind 1; QACC 0; A100-PCIE-40GB; init id and first
qpos equal to the already-collected pf_*_sp/pf_sham_cg same key; logs: Traceback 0, outcome lines 0.
Prints NO success, NO episode length, NO per-step array shape. Exit 1 on any failure."""
import glob, json, os, re, subprocess, sys
import numpy as np
H = os.path.expanduser; Z = H("$PROJ_ROOT/vla_zoo")
CELLS = ["pf_sham_cg", "pf_grpkp20", "pf_g_tfc_kp20"]
REF = {"rs_smoke_oftc_sp": "pf_oftc_sp", "rs_smoke_ofti_sp": "pf_ofti_sp"}
bad = []
def chk(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg); cond or bad.append(msg)
def want_env(cell):
    v = subprocess.check_output(["bash", f"{Z}/next/cells_next.sh", "vars", cell], text=True).split()
    d = {"BODY": v[0], "EMBODY_DIR": H("$PROJ_ROOT/embody_next"), "M1_FASTSYNC": "1"}
    d.update(dict(x.split("=", 1) for x in v[1:])); return d
for m, ref in REF.items():
    for c in CELLS:
        fs = sorted(glob.glob(f"{Z}/runs/{m}/{c}/t*_e*_i*.npz"))
        keys = sorted((int(a), int(b)) for a, b in (re.search(r"t(\d+)_e(\d+)_", f).groups() for f in fs))
        chk(keys == [(0, 0), (0, 1)], f"{m}/{c} keys {keys}")
        we = want_env(c)
        for f in fs:
            z = np.load(f, allow_pickle=True); k = os.path.basename(f)[:5]
            env = json.loads(str(z["pf_cell_env"]))
            chk(env == we, f"{m}/{c}/{k} cell_env {env}")
            chk(int(z["pf_blind"]) == 1 and int(z["pf_qacc_count"]) == 0, f"{m}/{c}/{k} blind=1 qacc=0")
            chk("A100-PCIE-40GB" in str(z["gpu"]), f"{m}/{c}/{k} gpu {z['gpu']}")
            r85 = "gripper0_finger_joint" in [str(x) for x in z["joint_names"]]
            na = np.asarray(z["pf_grip_armature"]).size
            chk(r85 == (we["BODY"] == "PandaR85") and na == (6 if r85 else 0), f"{m}/{c}/{k} body r85={r85} armature_dofs={na}")
            tw = "twin_stale" in z.files
            chk(tw == (we.get("TWIN_RENDER") == "1"), f"{m}/{c}/{k} twin fields present={tw}")
            if tw:
                cm, pm_ = float(np.nanmax(z["twin_cam_err_mm"])), float(np.nanmax(z["twin_pos_err_mm"]))
                chk(int(z["twin_stale"]) == 0 and cm < 1e-3 and pm_ < 1.0, f"{m}/{c}/{k} twin stale={int(z['twin_stale'])} cam_err_max={cm:.2e}mm ik_pos_max={pm_:.4f}mm")
            al = json.loads(str(z["align_json"]))
            chk(al.get("copy_grip_cmd") == 1 and al["pos_mm"] < 0.01, f"{m}/{c}/{k} align pos {al['pos_mm']:.4f}mm copy_grip_cmd {al.get('copy_grip_cmd')}")
            rf = glob.glob(f"{Z}/runs/{ref}/pf_sham_cg/{k}_i*.npz")
            same_init = len(rf) == 1 and os.path.basename(rf[0]) == os.path.basename(f)
            chk(same_init, f"{m}/{c}/{k} init id same as {ref}/pf_sham_cg ({os.path.basename(f)})")
            if same_init and c == "pf_sham_cg":
                q0, q1 = np.asarray(z["qpos"])[0], np.asarray(np.load(rf[0], allow_pickle=True)["qpos"])[0]
                chk(q0.shape == q1.shape and float(np.abs(q0 - q1).max()) == 0.0, f"{m}/{c}/{k} first qpos == {ref} S")
logs = []
for j in os.environ.get("SMOKE_JOBS", "").split(","):
    for lg in glob.glob(f"{Z}/logs/pf_rs_smoke_*_{j}.log"):
        t = open(lg, errors="replace").read()
        chk(t.count("Traceback") == 0 and "PF_LAUNCH blind=1" in t, f"{os.path.basename(lg)} Traceback={t.count('Traceback')} launch_line={'PF_LAUNCH blind=1' in t}")
        chk(len(re.findall(r"(?i)success", t)) == 0, f"{os.path.basename(lg)} outcome lines 0")
        logs.append(lg)
chk(len(logs) == 6, f"{len(logs)} smoke run logs (want 6)")
print("SMOKE", "FAIL" if bad else "PASS", len(bad)); sys.exit(1 if bad else 0)
