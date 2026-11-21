"""Check the runner self-test dumps (field presence / shapes / hook self-checks). Never reads success."""
import glob, json, sys
import numpy as np
root = sys.argv[1]
need = ["action", "eef_pos", "qpos", "mr_steps", "mr_checks", "mi_rows", "mi_spec", "mi_obj_names", "mi_obj_pos",
        "mi_grip_pos", "mi_contact_lpad", "mi_contact_rpad", "mi_contact_grip_target", "mi_contact_target_support",
        "mi_contact_grip_interactive", "mi_pad_geoms", "mi_block", "mi_protect_names", "gpu", "align_json"]
bad = 0
for f in sorted(glob.glob(f"{root}/*/C*/t*_e*_i*.npz")):
    z = np.load(f, allow_pickle=True)
    miss = [k for k in need if k not in z.files]
    T = len(z["action"])
    shp = {k: z[k].shape[0] for k in ("mi_obj_pos", "mi_grip_pos", "mi_contact_lpad", "mi_contact_target_support")}
    rows = json.loads(str(z["mi_rows"]))
    chk = json.loads(str(z["mr_checks"]))
    spec = json.loads(str(z["mi_spec"]))
    selfbad = sum(1 for r in rows for v, x in r.items() if x.get("out_E_diff_px"))
    md5bad = sum(1 for r in chk if r.get("model_input_is_edited") is False)
    sub = sum(1 for r in chk for v in ("agent", "wrist") if r.get(v, {}).get("same_substep") is False)
    ok = not miss and all(n == T + 1 for n in shp.values()) and len(rows) == len(z["mr_steps"]) and not selfbad and not md5bad and not sub
    bad += not ok
    print(f.split(root)[1], "OK" if ok else "BAD", "T+1" if all(n == T + 1 for n in shp.values()) else shp, "missing", miss,
          "frames", len(rows), "outsideE", selfbad, "md5bad", md5bad, "substep_bad", sub, "cell", spec["cell"],
          "protect", spec["protect"], "P", json.loads(str(z["mi_protect_names"]))[:9],
          "pads", str(z["mi_pad_geoms"]), "contacts l/r/t/s", int(z["mi_contact_lpad"].sum()), int(z["mi_contact_rpad"].sum()),
          int(z["mi_contact_grip_target"].sum()), int(z["mi_contact_target_support"].sum()))
print("BAD", bad)
