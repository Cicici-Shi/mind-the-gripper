import glob
"""
run_libero_eval_toy.py: copy of $PROJ_ROOT/openvla-oft/experiments/robot/libero/run_libero_eval.py (2026-11-03) for the
toy1 runs in $PROJ_ROOT/embody_toy. Differences: EMBODY_DIR defaults to $PROJ_ROOT/embody_toy; the dump also stores
toy_px (BodyEnv.toy_stats["px"], TOY_PIXSTATS=1) and the TOY1 settings inside align_json; QUERY_DUMP=1 also stores
the raw (un-normalised, pre-binarisation) action chunk of every policy query (query_step, query_chunk);
TOY_EP_KEEP="i,j" runs only those episodes of the shard (others reset + skipped, scenes unchanged).
Run with cwd $PROJ_ROOT/embody_toy and PYTHONPATH=$PROJ_ROOT/openvla-oft so rollouts/ and logs land in embody_toy.

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

try:  # NUS-PATCH: wandb 与本环境的 protobuf 3.20 冲突；本项目不用它，改为可选
    import wandb
except Exception as _e:
    wandb = None
    print(f'[NUS-PATCH] wandb 不可用（{type(_e).__name__}），已跳过；use_wandb 必须为 False')

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


# NUS-PATCH DUMP 2026-10-26: per-step trace for replay (RQ3) and stage metrics; on when DUMP_DIR is set
_DUMP_CTX = {}


_HOST_GPU = None


def _host_gpu():
    """NUS-PATCH 2026-10-31 (block 3 amendment 1, R8/R10): node and GPU model, recorded in every dump."""
    global _HOST_GPU
    if _HOST_GPU is None:
        import socket
        import subprocess
        gpu = os.environ.get("NUS_GPU", "")
        if not gpu:
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                     text=True, timeout=30).stdout.strip().splitlines()[0]
            except Exception as e:  # never break an episode for this
                gpu = f"unknown ({e!r})"
        _HOST_GPU = (socket.gethostname(), gpu)
    return _HOST_GPU


def _dump_episode(trace, success, twin=None, contacts=None, align_info=None, toy=None):
    d = os.environ.get("DUMP_DIR")
    if not d or not trace["action"]:
        return
    os.makedirs(d, exist_ok=True)
    c = _DUMP_CTX
    import json as _json
    _h, _g = _host_gpu()
    np.savez_compressed(
        os.path.join(d, f"t{c.get('task')}_e{c.get('ep')}_i{c.get('init')}.npz"),
        success=success, task=c.get("task"), ep=c.get("ep"), init=c.get("init"),
        action=np.asarray(trace["action"], dtype=np.float32),
        eef_pos=np.asarray(trace["eef_pos"], dtype=np.float32),
        eef_quat=np.asarray(trace["eef_quat"], dtype=np.float32),
        gripper_qpos=np.asarray(trace["gripper_qpos"], dtype=np.float32),
        qpos=np.asarray(trace["qpos"], dtype=np.float64),
        joint_names=np.asarray(trace["joint_names"]),
        ref_body_names=np.asarray(trace["ref_body_names"]),
        ref_body_pos=trace["ref_body_pos"], ref_body_quat=trace["ref_body_quat"],
        **(contacts or {}),  # NUS-PATCH 2026-10-27: online arm-environment contacts (embody_v3 contacts.py)
        host=_h, gpu=_g, align_json=_json.dumps(align_info),  # NUS-PATCH 2026-10-31
        **({} if not (toy and toy.get("px")) else {"toy_px": np.asarray(toy["px"], dtype=np.int32)}),  # TOY1
        **({} if not (twin and twin.get("cam_err_mm")) else {  # TOY1: twin vs physical wrist camera, per paint
            "twin_cam_err_mm": np.asarray(twin["cam_err_mm"], dtype=np.float32),
            "twin_cam_err_deg": np.asarray(twin["cam_err_deg"], dtype=np.float32)}),
        **({} if not trace.get("query_step") else {"query_step": np.asarray(trace["query_step"], dtype=np.int32),
                                                    "query_chunk": np.asarray(trace["query_chunk"], dtype=np.float32)}),
        **({} if not (twin and twin.get("root_off_mm")) else {  # NUS-PATCH 2026-10-31: G_R root-body offset per frame
            "twin_root_off_mm": np.asarray(twin["root_off_mm"], dtype=np.float32)}),
        **({} if not twin else dict(  # TWIN_RENDER checks (embody BodyEnv.twin_stats)
            twin_pos_err_mm=np.asarray(twin["pos_err_mm"], dtype=np.float32),
            twin_rot_err_deg=np.asarray(twin["rot_err_deg"], dtype=np.float32),
            twin_posture_deg=np.asarray(twin["posture_deg"], dtype=np.float32),
            twin_stale=twin["stale"],
            twin_t0=np.asarray([v for k in ("agentview_image", "robot0_eye_in_hand_image") for v in twin["t0"][k]],
                               dtype=np.float32) if twin.get("t0") else np.zeros(4, np.float32),
        )),
    )


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    # NUS-PATCH: BodyEnv 已在参考 Panda 上完成等待步，策略直接开始，步数预算不变
    t = cfg.num_steps_wait if getattr(env, "skip_wait", False) else 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    _m = env.sim.model
    _rm = (env.ref if hasattr(env, "ref") else env).sim.model  # reference (Panda) scene, for cross-run scene checks
    trace = {"action": [], "eef_pos": [], "eef_quat": [], "gripper_qpos": [], "qpos": [],
             "joint_names": [_m.joint_id2name(j) or "" for j in range(_m.njnt)],
             "ref_body_names": [_rm.body_id2name(b) or "" for b in range(_rm.nbody)],
             "ref_body_pos": _rm.body_pos.copy(), "ref_body_quat": _rm.body_quat.copy()}

    # NUS-PATCH 2026-10-27: online arm-environment contact log, created at the first policy step
    try:
        from embody import contacts as _contacts
    except ImportError:
        _contacts = None
    _clog, _clog_err = None, None

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
                if os.environ.get("QUERY_DUMP") == "1":  # TOY1: raw chunk, before process_action
                    trace.setdefault("query_step", []).append(len(trace["action"]))
                    trace.setdefault("query_chunk", []).append(np.asarray(actions, dtype=np.float32))

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            trace["action"].append(np.asarray(action, dtype=np.float32))
            trace["eef_pos"].append(obs["robot0_eef_pos"].copy())
            trace["eef_quat"].append(obs["robot0_eef_quat"].copy())
            trace["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
            trace["qpos"].append(env.sim.data.qpos.copy())
            if _clog is None and _contacts is not None and _clog_err is None:
                try:
                    _clog = _contacts.make(env)
                except Exception as _ce:  # never let the contact log break an episode
                    _clog_err = repr(_ce)
                    log_message(f"CONTACT_LOG error: {_clog_err}", log_file)
            obs, reward, done, info = env.step(action.tolist())
            if _clog is not None:
                try:
                    _clog.update(action)
                except Exception as _ce:
                    _clog_err, _clog = repr(_ce), None
                    log_message(f"CONTACT_LOG error: {_clog_err}", log_file)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        import traceback as _tb
        log_message(f"Episode error: {e}\n{_tb.format_exc()}", log_file)

    try:
        _dump_episode(trace, success, getattr(env, "twin_stats", None),
                      _clog.npz_fields() if _clog is not None else ({"contact_error": _clog_err} if _clog_err else None),
                      getattr(env, "align_info", None), getattr(env, "toy_stats", None))
    except Exception as e:
        log_message(f"Dump error: {e}", log_file)
    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    # NUS-PATCH: BODY=Panda|IIWA|UR5e 时在另一本体上跑（Panda 即 SRC-sham），见 $PROJ_ROOT/embody
    if os.environ.get("BODY"):
        import sys as _sys; _sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_toy")))
        from embody.body_env import BodyEnv
        from libero.libero import get_libero_path
        env = BodyEnv(env, os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                      os.environ["BODY"], cfg.env_img_res, cfg.num_steps_wait,
                      log=lambda m: log_message(m, log_file))

    # Start episodes
    task_episodes, task_successes = 0, 0
    # NUS-PATCH 2026-10-26: shards run episodes [EP_OFFSET, EP_OFFSET + num_trials_per_task) of the fixed permutation
    _ep0 = int(os.environ.get("EP_OFFSET", "0"))
    for episode_idx in tqdm.tqdm(range(_ep0, _ep0 + cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            # NUS-PATCH: 按固定种子预抽初始状态编号，SRC 前 100 条可作为正式 200 条的前一半
            _perm = np.random.RandomState(5478 + task_id).permutation(len(initial_states))
            _init_id = int(_perm[episode_idx])
            log_message(f"INIT_ID: task={task_id} episode={episode_idx} init={_init_id}", log_file)
            _DUMP_CTX.update(task=task_id, ep=episode_idx, init=_init_id)
            initial_state = initial_states[_init_id]
            # NUS-PATCH 2026-10-26: SKIP_DONE=1 fills holes in a shard. The env is still reset so the
            # global np.random stream (fixture placement) advances exactly as in a full shard run.
            # TOY1: TOY_EP_KEEP="0,1" runs only these episode indices of the shard; the others are reset and skipped so
            # the global np.random stream (fixtures) advances exactly as in a full shard (same scenes as published cells)
            _keep = os.environ.get("TOY_EP_KEEP", "")
            if _keep and episode_idx not in {int(x) for x in _keep.split(",")}:
                env.reset()
                log_message(f"TOY_EP_KEEP: task={task_id} episode={episode_idx} skipped", log_file)
                continue
            if os.environ.get("SKIP_DONE") == "1" and os.environ.get("DUMP_DIR") and glob.glob(
                    os.path.join(os.environ["DUMP_DIR"], f"t{task_id}_e{episode_idx}_i*.npz")):
                env.reset()
                log_message(f"SKIP_DONE: task={task_id} episode={episode_idx} already dumped", log_file)
                continue
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # NUS-PATCH: MAX_TASKS 限制任务数；FILTER_CATEGORY 按 LIBERO-Plus 的扰动类别筛选，
    # 这样才能和它成绩表里按类别报的分数逐项对比。
    import os as _os, json as _json
    _ids = list(range(num_tasks))
    _cat = _os.environ.get("FILTER_CATEGORY")
    if _cat:
        _tc = _json.load(open(_os.path.expanduser(
            "$PROJ_ROOT/libero-plus/libero/libero/benchmark/task_classification.json")))
        _items = _tc[cfg.task_suite_name]
        _keep = {i for i, it in enumerate(_items) if it.get("category") == _cat}
        _ids = [i for i in _ids if i in _keep]
        print(f"[NUS-PATCH] 类别 '{_cat}' 共 {len(_ids)} 个任务")
    # NUS-PATCH: 检查分类表下标与任务名是否一一对应（Stage 1 按下标筛类别，此前未核验）
    _tc_all = _json.load(open(_os.path.expanduser(
        "$PROJ_ROOT/libero-plus/libero/libero/benchmark/task_classification.json")))[cfg.task_suite_name]
    _mis = [i for i, it in enumerate(_tc_all) if i < num_tasks and task_suite.get_task(i).name != it["name"]]
    print(f"[NUS-PATCH] 分类表 {len(_tc_all)} 项, 任务 {num_tasks} 个, 下标与名字不符 {len(_mis)} 处, 例: {_mis[:5]}")
    if _os.environ.get("ORIGINAL_ONLY"):
        import sys as _sys; _sys.path.insert(0, _os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_toy")))
        from embody.original import OriginalSpatialSuite
        if cfg.task_suite_name == "libero_object":  # NUS-PATCH 2026-10-27: block 3 second task suite (embody_v3)
            from embody.original import OriginalObjectSuite
            task_suite = OriginalObjectSuite()
        else:
            task_suite = OriginalSpatialSuite()
        _ids = list(range(task_suite.n_tasks))
        print(f"[NUS-PATCH] 无扰动原始任务 {len(_ids)} 个: {[task_suite.get_task(i).name for i in _ids]}")
    _mt = _os.environ.get("MAX_TASKS")
    if _mt:
        _ids = _ids[: int(_mt)]
        print(f"[NUS-PATCH] 截断为前 {len(_ids)} 个任务")
    for task_id in tqdm.tqdm(_ids):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
