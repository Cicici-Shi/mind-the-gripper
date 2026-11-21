"""maskinfo-pi (2026-11-08, protocol_maskinfo_pi.md v2 2.2 / 2.4 / 2.6 item 11): openpi policy server with FIXED
flow-matching noise.

Same policy construction as $PROJ_ROOT/pi05/openpi/scripts/serve_policy.py (policy_config.create_trained_policy on
config + checkpoint dir; the openpi tree is only imported, never modified) and the same WebsocketPolicyServer. The
only change is the policy object handed to the server:

  noise = jax.random.normal(jax.random.key(noise_seed), (action_horizon, action_dim)), float32
  noise_seed comes with every request (client: int.from_bytes(sha256(f"{MASTER}|{model}|{suite}|{task}|{ep}|{q}")
  [:8], "big") % 2**63, protocol 2.4), together with mi_q (the query index). Both are removed before the openpi
  input transforms. Requests without noise_seed are refused (no fall back to the server's rng stream).

Every response carries mi_noise = {q, seed, md5 of the float32 noise bytes}. At the first request of an episode
(mi_q == 0) the received 8-d state is appended to --state-log (jsonl, one line per episode, formal runs too;
protocol 2.2 / gate item 14) and echoed as mi_S_server.received. --debug-state-transform (end-to-end check runs
only, OFF in formal runs) additionally runs the policy's own input transforms on a copy of that first request and
returns the transformed state tensor (mi_S_server.transformed) for the comparison with the expected transform.
The effective configuration is printed at start (MI_SERVER_CFG) and published in the metadata (mi_server_cfg); the
state-path assertions (pi0.5: no continuous state token, discrete_state_input False) and the three image masks of
the first request are logged once (MI_STATE_PATH, MI_IMAGE_MASK).
Usage (cwd $PROJ_ROOT/pi05/openpi, its .venv python):
  python serve_policy_mi.py --config pi05_libero --dir <ckpt> --port P --state-log FILE [--debug-state-transform]
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import socket
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--state-log", required=True)
    ap.add_argument("--debug-state-transform", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    import jax
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config
    from openpi_client import base_policy as _base_policy

    cfg = _config.get_config(a.config)
    policy = _policy_config.create_trained_policy(cfg, a.dir)
    model = policy._model
    H, D = int(model.action_horizon), int(model.action_dim)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    mc = cfg.model
    norms = []  # the Normalize / Unnormalize actually inside the policy (stats loaded from the checkpoint assets)
    for side, comp in (("in", policy._input_transform), ("out", policy._output_transform)):
        for t in getattr(comp, "transforms", []):
            if type(t).__name__ in ("Normalize", "Unnormalize"):
                norms.append({"side": side, "type": type(t).__name__, "use_quantiles": bool(getattr(t, "use_quantiles", False)),
                              "keys": sorted((getattr(t, "norm_stats", None) or {}).keys())})
    eff = {
        "config": a.config, "ckpt_dir": os.path.abspath(a.dir),
        "model_class": type(model).__name__, "model_type": str(getattr(mc, "model_type", "")),
        "pi05": bool(getattr(mc, "pi05", False)), "discrete_state_input": bool(getattr(mc, "discrete_state_input", False)),
        "action_horizon": H, "action_dim": D, "max_token_len": int(getattr(mc, "max_token_len", -1)),
        "paligemma_variant": str(getattr(mc, "paligemma_variant", "")),
        "action_expert_variant": str(getattr(mc, "action_expert_variant", "")),
        "use_quantile_norm": bool(dc.use_quantile_norm), "policy_norm_transforms": norms,
        "prompt_from_task": bool(getattr(dc, "prompt_from_task", False)),
        "extra_delta_transform": bool(getattr(cfg.data, "extra_delta_transform", False)),
        "input_transforms": [type(t).__name__ for t in getattr(policy._input_transform, "transforms", [])],
        "output_transforms": [type(t).__name__ for t in getattr(policy._output_transform, "transforms", [])],
        "noise": {"shape": [H, D], "seed_bits_note": "jax x32: jax.random.key keeps the seed as given; see WORKLOG", "dtype": "float32", "draw": "jax.random.normal(jax.random.key(noise_seed), shape)",
                  "seed": "client: int.from_bytes(sha256(f'{MASTER}|{model}|{suite}|{task}|{ep}|{q}')[:8], 'big') % 2**63"},
        "debug_state_transform": bool(a.debug_state_transform), "state_log": os.path.abspath(a.state_log),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "host": socket.gethostname(), "gpu": os.environ.get("NUS_GPU", ""),
    }
    print("MI_SERVER_CFG " + json.dumps(eff), flush=True)
    tok = [t for t in getattr(policy._input_transform, "transforms", []) if type(t).__name__ == "TokenizePrompt"]
    tok_dsi = [bool(getattr(t, "discrete_state_input", False)) for t in tok]
    if eff["pi05"]:  # protocol 2.2: pi0.5 has neither state path (config value + runtime assertion, one log line each)
        assert not eff["discrete_state_input"], "pi0.5 config discrete_state_input: state would enter the prompt"
        assert bool(getattr(model, "pi05", False)), "loaded model is not pi05: it would add a continuous state token"
        assert tok and not any(tok_dsi), f"runtime TokenizePrompt discrete_state_input {tok_dsi}"
        print("MI_STATE_PATH pi05 config: discrete_state_input=False, pi05=True; runtime: loaded model.pi05=True "
              f"(pi0.py adds no continuous state token), TokenizePrompt.discrete_state_input={tok_dsi} (asserted)", flush=True)
    else:
        print("MI_STATE_PATH pi0 config: continuous state token (pi05=False), discrete_state_input="
              f"{eff['discrete_state_input']}; runtime: model.pi05={bool(getattr(model, 'pi05', False))}, "
              f"TokenizePrompt.discrete_state_input={tok_dsi}", flush=True)
    slog = open(a.state_log, "a")
    seen = {"mask": False}

    class FixedNoisePolicy(_base_policy.BasePolicy):
        def __init__(self, inner):
            self._inner = inner

        def infer(self, obs):
            seed, q = obs.pop("noise_seed", None), obs.pop("mi_q", None)
            if seed is None or q is None:
                raise ValueError("maskinfo-pi server: request without noise_seed / mi_q refused")
            seed, q = int(seed), int(q)
            nz = np.asarray(jax.random.normal(jax.random.key(seed), (H, D)), dtype=np.float32)
            s_rec = None
            if q == 0:
                st = np.asarray(obs.get("observation/state"), np.float64) if "observation/state" in obs else None
                s_rec = {"received": None if st is None else st.tolist(), "transformed": None}
                slog.write(json.dumps({"noise_seed": seed, "state": s_rec["received"]}) + "\n")  # no time (blind)
                slog.flush()
                if a.debug_state_transform or not seen["mask"]:
                    tr = self._inner._input_transform(copy.deepcopy(obs))  # the policy's own transforms, on a copy
                    if a.debug_state_transform:
                        s_rec["transformed"] = np.asarray(tr["state"], np.float32).tolist()
                    if not seen["mask"]:
                        print("MI_IMAGE_MASK " + json.dumps({k: bool(np.asarray(v).item()) if np.asarray(v).size == 1
                                                             else np.asarray(v).tolist()
                                                             for k, v in tr.get("image_mask", {}).items()}), flush=True)
                        seen["mask"] = True
            out = self._inner.infer(obs, noise=nz)
            out["mi_noise"] = {"q": q, "seed": seed, "md5": hashlib.md5(nz.tobytes()).hexdigest()}
            if s_rec is not None:
                out["mi_S_server"] = s_rec
            return out

        @property
        def metadata(self):
            return self._inner.metadata

    md = dict(policy.metadata or {})
    md["mi_server_cfg"] = eff
    server = websocket_policy_server.WebsocketPolicyServer(policy=FixedNoisePolicy(policy), host="0.0.0.0",
                                                           port=a.port, metadata=md)
    server.serve_forever()


if __name__ == "__main__":
    main()
