"""Does pi05_libero use the 8-d proprio state?

Loads the policy in-process (openpi env), takes observations dumped by pi05_libero_eval.py
(--dump-obs), and queries each one with identical images/prompt and an identical, explicitly
passed flow-matching noise tensor (np.random.default_rng(0), shape [action_horizon=10, action_dim=32]),
while varying only the state vector. Policy.infer(obs, noise=...) forwards the noise to
Pi0.sample_actions, which then never draws from the JAX rng.

Controls: same input twice (determinism), and a different noise seed with the real state
(how much the actions move when something the model does use changes).
"""
import glob
import json
import os
import sys

import numpy as np
from openpi.policies import policy_config
from openpi.training import config as _config

ckpt = sys.argv[1]
obs_dir = sys.argv[2]
out = sys.argv[3]
max_obs = int(sys.argv[4]) if len(sys.argv) > 4 else 8

cfg = _config.get_config("pi05_libero")
policy = policy_config.create_trained_policy(cfg, ckpt)
H, D = cfg.model.action_horizon, cfg.model.action_dim
noise0 = np.random.default_rng(0).standard_normal((H, D)).astype(np.float32)
noise1 = np.random.default_rng(1).standard_normal((H, D)).astype(np.float32)

files = sorted(glob.glob(os.path.join(obs_dir, "*.npz")))[:max_obs]
rows = []
for fpath in files:
    z = np.load(fpath)
    base = {"observation/image": z["image"], "observation/wrist_image": z["wrist_image"], "prompt": str(z["prompt"])}
    s = z["state"].astype(np.float64)

    def run(state, noise):
        return np.asarray(policy.infer({**base, "observation/state": state}, noise=noise)["actions"], dtype=np.float64)

    ref = run(s, noise0)
    variants = {
        "repeat_same": (s, noise0),
        "zeros": (np.zeros_like(s), noise0),
        "eef_x+5cm": (s + np.array([0.05, 0, 0, 0, 0, 0, 0, 0]), noise0),
        "eef_z+10cm": (s + np.array([0, 0, 0.10, 0, 0, 0, 0, 0]), noise0),
        "gripper_flip": (np.concatenate([s[:6], -s[6:]]), noise0),
        "random_state": (np.random.default_rng(2).normal(0, 1, 8), noise0),
        "CONTROL_other_noise": (s, noise1),
    }
    row = {"obs": os.path.basename(fpath), "state": s.round(4).tolist(), "ref_action0": ref[0].round(4).tolist()}
    for k, (st, nz) in variants.items():
        a = run(st, nz)
        row[k] = {"max_abs": float(np.abs(a - ref).max()), "mean_abs": float(np.abs(a - ref).mean())}
    rows.append(row)
    print(json.dumps(row), flush=True)

with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
print("SUMMARY over", len(rows), "observations (actions: 10 x 7, unnormalized env units)")
for k in keys:
    mx = max(r[k]["max_abs"] for r in rows)
    mn = np.mean([r[k]["mean_abs"] for r in rows])
    print(f"  {k:22s} max|da|={mx:.3e}  mean|da|={mn:.3e}")
