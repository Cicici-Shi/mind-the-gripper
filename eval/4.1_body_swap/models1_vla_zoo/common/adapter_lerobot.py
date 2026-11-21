"""vla_zoo adapter: any LeRobot (0.6.1) LIBERO policy checkpoint, e.g. HuggingFaceVLA/smolvla_libero, GR00T, xvla.

Reproduces lerobot/scripts/lerobot_eval.py::rollout for one env (no re-implementation of preprocessing):
  raw env obs -> LiberoEnv._format_raw_obs layout {"pixels": {image, image2}, "robot_state": {eef, gripper, joints}}
  -> envs.utils.preprocess_observation -> observation["task"] = [prompt]
  -> env_preprocessor = PolicyProcessorPipeline([LiberoProcessorStep()])  (180-deg flip, 8-d state from quat->axis-angle)
  -> policy preprocessor (from the checkpoint) -> policy.select_action -> postprocessor (unnormalize).
Images arrive exactly as rendered by the oftpi client (256x256, not flipped), same as LiberoEnv(observation 256).
Actions: LeRobot passes them to LIBERO unchanged -> already LIBERO convention.
replan: the policy's own n_action_steps (select_action pops its queue); we return the whole queue once per query so
the client executes n_action_steps actions open loop, identical to select_action being called every step.
"""
import os

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
from lerobot.configs.policies import PreTrainedConfig


class Adapter:
    def __init__(self, repo="HuggingFaceVLA/smolvla_libero", n_action_steps="", device="cuda", cam1="image", cam2="image2", **kw):
        cfg = PreTrainedConfig.from_pretrained(repo)
        for k, v in kw.items():   # cfg_<attr>=<value>: override a config field (e.g. GR00T base_model_path)
            assert k.startswith("cfg_"), k
            setattr(cfg, k[4:], os.path.expanduser(v))
        if n_action_steps:
            cfg.n_action_steps = int(n_action_steps)
        cfg.device = device
        cls = get_policy_class(cfg.type)
        self.policy = cls.from_pretrained(repo, config=cfg)
        self.overrides = {k: v for k, v in kw.items()}
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=repo,
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.cam1, self.cam2 = cam1, cam2
        self.n = int(cfg.n_action_steps)
        self.metadata = {"model": repo, "type": cfg.type, "replan": self.n, "chunk": int(getattr(cfg, "chunk_size", self.n)),
                         "cams": "agentview+wrist", "proprio": True, "lerobot": __import__("lerobot").__version__,
                         "cams_keys": [cam1, cam2], "overrides": dict(kw)}

    @torch.inference_mode()
    def infer(self, o):
        raw = {
            "pixels": {self.cam1: np.asarray(o["agentview"])[None], self.cam2: np.asarray(o["wrist"])[None]},
            "robot_state": {
                "eef": {"pos": np.asarray(o["eef_pos"], np.float64)[None], "quat": np.asarray(o["eef_quat"], np.float64)[None]},
                "gripper": {"qpos": np.asarray(o["gripper_qpos"], np.float64)[None]},
                "joints": {"pos": np.asarray(o["joint_pos"], np.float64)[None]},
            },
        }
        obs = preprocess_observation(raw)
        obs["task"] = [str(o["prompt"])]
        obs = self.env_pre(obs)
        obs = self.pre(obs)
        self.policy.reset()                     # fresh queue: next select_action computes a new chunk
        acts = []
        for i in range(self.n):                 # pops the n queued actions of that one chunk (no new inference)
            a = self.policy.select_action(obs)
            acts.append(self.post(a).float().cpu().numpy().reshape(-1))
        return np.stack(acts).astype(np.float64)
