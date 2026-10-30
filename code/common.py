import os

os.environ["MUJOCO_GL"] = "egl" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from pathlib import Path

import jax
import numpy as np
import yaml
from jax import numpy as jp
import mediapy as media

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VIDEOS = ROOT / "videos"
CHECKPOINTS = ROOT / "checkpoints"

for directory in (RESULTS, VIDEOS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

with open(ROOT / "configs" / "config.yaml") as file:
    CFG = yaml.safe_load(file)

ENV_NAME = CFG["env_name"]
SUCCESS_THRESH = float(CFG["success_threshold"])
TEACHER_CKPT_DIR = ROOT / CFG["teacher"]["checkpoint_dir"]


env_cfg = registry.get_default_config(ENV_NAME)
env_cfg.impl = CFG["impl"]
env = registry.load(ENV_NAME, config=env_cfg)


def box_pos(state):
    """Return the cube position from forward kinematics.

    This should only be used after a physics step because data.xpos may not be
    populated immediately after reset.
    """
    return state.data.xpos[env._obj_body]


def box_pos_from_qpos(state):
    """Return the cube position directly from its free-joint coordinates.

    This is valid immediately after reset.
    """
    start = env._obj_qposadr
    return state.data.qpos[start:start + 3]


def dist_to_target(state):
    """Return the cube-center to target distance in metres."""
    delta = box_pos(state) - state.info["target_pos"]
    return float(jp.linalg.norm(delta))

def success_times(dists, thresh):
    within = np.asarray(dists) < thresh

    if not within.any():
        return -1, -1

    first_reached_step = int(np.flatnonzero(within)[0]) + 1

    if not within[-1]:
        return first_reached_step, -1

    outside = np.flatnonzero(~within)
    final_start_index = int(outside[-1]) + 1 if len(outside) else 0
    final_success_start_step = final_start_index + 1

    return first_reached_step, final_success_start_step

def ppo_params():
    """Playground's default PPO hyperparameters for this env."""
    return manipulation_params.brax_ppo_config(ENV_NAME)


def latest_checkpoint(ckpt_dir):
    steps = [d for d in os.listdir(ckpt_dir) if d.isdigit()]
    return Path(ckpt_dir) / max(steps, key=int)


def load_teacher(path, deterministic=True):
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ppo_params().network_factory,
    )
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    params = ppo_checkpoint.load(str(path))
    return jax.jit(make_inference_fn(params, deterministic=deterministic))

def save_video(states, path, height=480, width=640):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = env.render(states, height=height, width=width)
    media.write_video(str(path), frames, fps=round(1.0 / env.dt))