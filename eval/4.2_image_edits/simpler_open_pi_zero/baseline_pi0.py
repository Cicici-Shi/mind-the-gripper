# 问题 4：open-pi-zero Bridge-Beta 在 SimplerEnv 一个任务上跑 N 回合原样基线（不改画面）。
# 循环照抄 open-pi-zero/src/agent/eval.py：episode_id 依次取 0..N-1，截断时取 success。
import os, sys, time, json, argparse
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/bench2_simpler/open-pi-zero"))
os.chdir(os.path.expanduser("$PROJ_ROOT/bench2_simpler/open-pi-zero"))
import numpy as np, torch, hydra, imageio, simpler_env
from omegaconf import OmegaConf
from src.model.vla.pizero import PiZeroInference

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--bf16", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
cfg = OmegaConf.load("config/eval/bridge.yaml" if "bridge" in args.ckpt else "config/eval/fractal_apple.yaml")
dev = torch.device("cuda:0"); dtype = torch.bfloat16 if args.bf16 else torch.float32
t0 = time.time()
model = PiZeroInference(cfg, use_ddp=False)
data = torch.load(args.ckpt, weights_only=True, map_location="cpu")
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}, strict=True)
del data
model.freeze_all_weights(); model.to(dtype); model.to(dev); model.eval()
print("model loaded", round(time.time() - t0, 1), "s", flush=True)
env = simpler_env.make(args.task)
adapter = hydra.utils.instantiate(cfg.env.adapter)
res = []
for ep in range(args.n):
    te = time.time()
    obs, info = env.reset(options={"obj_init_options": {"episode_id": ep}})
    adapter.reset(); instr = env.get_language_instruction()
    frames = [adapter.get_video_frame(env, obs)]; n_inf = 0; t_inf = 0.0; steps = 0
    while True:
        inp = adapter.preprocess(env, obs, instr)
        cm, vp, pp, apos = model.build_causal_mask_and_position_ids(inp["attention_mask"], dtype=dtype)
        itpm, am = model.split_full_mask_into_submasks(cm)
        x = {"input_ids": inp["input_ids"], "pixel_values": inp["pixel_values"].to(dtype),
             "image_text_proprio_mask": itpm, "action_mask": am, "vlm_position_ids": vp,
             "proprio_position_ids": pp, "action_position_ids": apos, "proprios": inp["proprios"].to(dtype)}
        x = {k: v.to(dev) for k, v in x.items()}
        ti = time.time()
        with torch.inference_mode():
            acts = model(**x)
        torch.cuda.synchronize(); t_inf += time.time() - ti; n_inf += 1
        for a in adapter.postprocess(acts[0].float().cpu().numpy())[: cfg.act_steps]:
            obs, r, success, truncated, info = env.step(a); steps += 1
            if truncated: break
        frames.append(adapter.get_video_frame(env, obs))
        new = env.get_language_instruction()
        if new != instr: instr = new
        if truncated: break
    rec = dict(ep=ep, success=bool(success), steps=steps, sec=round(time.time() - te, 2),
               inf_calls=n_inf, inf_sec_mean=round(t_inf / n_inf, 4), instruction=instr)
    print(json.dumps(rec), flush=True); res.append(rec)
    imageio.mimsave(os.path.join(args.out, f"ep{ep}_{'S' if success else 'F'}.mp4"), frames, fps=5)
print("SUMMARY", args.task, "success", sum(r["success"] for r in res), "/", len(res),
      "mean sec/ep", round(np.mean([r["sec"] for r in res]), 2),
      "peak VRAM GB", round(torch.cuda.max_memory_reserved() / 1024**3, 2), flush=True)
json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
