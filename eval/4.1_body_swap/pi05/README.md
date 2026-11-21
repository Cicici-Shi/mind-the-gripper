# pi0.5-LIBERO on the 10 original LIBERO-spatial tasks (NUS SoC Slurm)

## What is here
- `openpi/`  Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (2026-08-24), uv venv `openpi/.venv`
  (Python 3.11.17, jax 0.5.3 [cuda12], torch 2.7.1). Built by `install_openpi.sbatch` (CPU job, ~27 min, mostly NFS file copy).
- `ckpt/pi05_libero/`  gs://openpi-assets/checkpoints/pi05_libero (16 files, 12,439,085,481 bytes), fetched over public
  HTTPS by `download_ckpt.sh` on the login node (~12 min). Tokenizer: `~/.cache/openpi/big_vision/paligemma_tokenizer.model`.
- conda env `oftpi` = clone of `oft` (mujoco 2.3.2, robosuite 1.4.1 incl. the EGL patches, numpy 1.26.4) + openpi-client
  (editable), websockets 16.1.1, msgpack. Built by `clone_env.sbatch`.
- `pi05_libero_eval.py`  websocket client (oftpi env). Same preprocessing as openpi `examples/libero/main.py`
  (180-degree flip, resize_with_pad 224, agentview + wrist, state = eef pos + axis-angle + 2 gripper qpos, replan every 5 of
  the 10 predicted actions, 10 wait steps, 220 max steps, render 256). Tasks from `embody.original.OriginalSpatialSuite`;
  init state = `RandomState(5478+task_id).permutation(50)[episode]`, env.seed(0), np.random.seed(7): identical to the OFT
  runs (checked: task0 ep0/1 -> init 42/21, task9 ep1 -> 28, same as ~/logs/cond_*.log).
- `state_sensitivity.py`  in-process policy (openpi env), varies only the state with fixed noise.
- `run_pi05.sbatch`  one GPU job: server in background, wait for port, client, kill server, sensitivity.

## Run
    cd $PROJ_ROOT/pi05
    PHASES=smoke,base,sens TRIALS=2 sbatch run_pi05.sbatch     # results_src${TRIALS}0.jsonl, state_sensitivity.jsonl
Client options: `--trials N --ep-offset K --tasks 0,3 --out file.jsonl --dump-obs dir`.
Server alone: `cd openpi && .venv/bin/python scripts/serve_policy.py --port P policy:checkpoint --policy.config=pi05_libero --policy.dir=$PROJ_ROOT/pi05/ckpt/pi05_libero`

## Results (job <job>, H100 NVL, 2026-10-26)
- Smoke: task 0, init 42, success in 77 steps.
- Baseline `results_src20.jsonl`: 20/20 success (100%); openpi reports 98.8% on LIBERO-Spatial (500 episodes).
  Mean 102 policy steps per episode.
- State sensitivity (`state_sensitivity.jsonl`, 12 dumped observations, query 0 and 10 of each episode): with identical
  images, prompt and noise, replacing the state by zeros / eef x+5 cm / eef z+10 cm / flipped gripper / random N(0,1)
  changes the 10x7 action chunk by exactly 0.0 (bitwise identical). Control: changing only the noise seed gives
  max |da| = 0.14, mean 0.011. Noise was fixed by passing `noise=` (np.random.default_rng(0), shape [10, 32], float32)
  to `Policy.infer`, which forwards it to `Pi0.sample_actions` (the JAX rng is then unused).
  This matches the code: config `pi05_libero` = `Pi0Config(pi05=True, discrete_state_input=False)`; with pi05=True
  `embed_suffix` adds no state token, and discrete_state_input=False keeps the state out of the prompt, and there is no
  delta-action transform (extra_delta_transform=False). The state is padded and normalized but never read by the network.

## Timings
- Server start to listening: 67 s (checkpoint restore 15 s + model build/JIT). First query ~20 s more (JIT of sample_actions).
- Per query round trip 67 ms (flow-matching 10 steps, batch 1). Per episode 20-34 s, mean 27.2 s wall (sim + EGL render
  dominate: ~0.27 s per env step including amortized queries). 20 episodes: 575 s.

## Pitfalls hit
- a100-40 is capped at 2 per user by QOS (`gres/gpu:a100-40=2`) and both were held by the main experiments, so this job
  used `--gres=gpu:h100-47:1` (got an H100 NVL); it does not touch the a100-40 quota. `sbatch --test-only` per gres
  type shows which one starts first. A short `--time` (50 min) helps backfill.
- Compute nodes do have internet (storage.googleapis.com reachable), so uv sync works in a CPU job.
- `nohup ... &` over ssh hangs the ssh session unless stdin is redirected (`< /dev/null`).
- The TCP port probe in the sbatch produces one harmless `InvalidMessage` traceback in the server log.
- libEGL "Permission denied" warnings for /dev/dri/* are harmless (rendering works).
