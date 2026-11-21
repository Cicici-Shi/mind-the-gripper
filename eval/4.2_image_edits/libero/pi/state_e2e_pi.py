"""maskinfo-pi end-to-end S check (astra_three_protocols_rulings_2026-11-08 §1 item 4), offline on the npz dumps.
For each npz: (1) the state the server received at the episode's first request == the 8-d state the client sent
(bitwise, recorded by the client as mi_S_server_eq_sent) and == the state rebuilt from the pre-edit observables
(mi_S_sent_ok[0]); (2) the server's transformed state tensor (the policy's own input transforms, recorded by
pi/serve_policy_mi.py) == the EXPECTED transform computed here independently from the checkpoint's
assets/physical-intelligence/libero/norm_stats.json: z-score (x - mean) / (std + 1e-6) or quantile
(x - q01) / (q99 - q01 + 1e-6) * 2 - 1 as the server reports, then zero padding to action_dim. Compared with
np.allclose(rtol=1e-6, atol=1e-6) (float32 vs float64 arithmetic); max |diff| printed. Never loads success.
Usage: python -I tools/state_e2e_pi.py NPZ...
"""
import json
import os
import sys

import numpy as np

bad = 0
for f in sys.argv[1:]:
    z = np.load(f, allow_pickle=True)
    cfg = json.loads(str(z["mi_server_cfg"]))
    norm = [t for t in cfg["policy_norm_transforms"] if t["side"] == "in" and t["type"] == "Normalize"]
    assert len(norm) == 1, norm
    stats = json.load(open(os.path.join(cfg["ckpt_dir"], "assets", "physical-intelligence", "libero", "norm_stats.json")))
    st = stats.get("norm_stats", stats)["state"]
    x = np.asarray(z["mi_S_server_received"], np.float64)
    if norm[0]["use_quantiles"]:
        q01, q99 = np.asarray(st["q01"])[: x.shape[-1]], np.asarray(st["q99"])[: x.shape[-1]]
        exp = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    else:
        mean, std = np.asarray(st["mean"])[: x.shape[-1]], np.asarray(st["std"])[: x.shape[-1]]
        exp = (x - mean) / (std + 1e-6)
    exp = np.pad(exp, (0, int(cfg["action_dim"]) - exp.shape[-1]))
    got = np.asarray(z["mi_S_server_transformed"], np.float64)
    ok_t = got.shape == exp.shape and np.allclose(got, exp, rtol=1e-6, atol=1e-6)
    ok_r = bool(np.asarray(z["mi_S_server_eq_sent"]).item()) and bool(np.asarray(z["mi_S_sent_ok"])[0])
    bad += not (ok_t and ok_r)
    print(f"S_E2E {os.path.relpath(f)} model={cfg['config']} quantile={norm[0]['use_quantiles']} received_eq_sent_eq_preedit={ok_r} "
          f"transformed_eq_expected={ok_t} max_abs_diff={float(np.abs(got - exp).max()) if got.shape == exp.shape else 'shape'} "
          f"state_token_path={'none (pi05, discrete_state_input False)' if cfg['pi05'] and not cfg['discrete_state_input'] else 'continuous state token'}")
print("S_E2E_BAD", bad)
sys.exit(1 if bad else 0)
