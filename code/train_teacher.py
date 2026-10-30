# Source: https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb#scrollTo=LcibXbyKt4FI

from common import (CFG, ENV_NAME, RESULTS, TEACHER_CKPT_DIR, env, ppo_params)

import functools
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

TRAIN_SEED = CFG["teacher"]["train_seed"]


if TEACHER_CKPT_DIR.exists() and any(TEACHER_CKPT_DIR.iterdir()):
    raise SystemExit(f"{TEACHER_CKPT_DIR} is not empty; delete it before retraining.")

TEACHER_CKPT_DIR.mkdir(parents=True, exist_ok=True)

params_cfg = ppo_params()
training_params = dict(params_cfg)

network_factory = ppo_networks.make_ppo_networks
if "network_factory" in params_cfg:
    del training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **params_cfg.network_factory
    )

x_data, y_data, y_err = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(int(num_steps))
    y_data.append(float(metrics["eval/episode_reward"]))
    y_err.append(float(metrics["eval/episode_reward_std"]))
    print(f"step {num_steps}: reward {y_data[-1]:.1f} ± {y_err[-1]:.1f}")


train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=network_factory,
    progress_fn=progress,
    seed=TRAIN_SEED,
    save_checkpoint_path=str(TEACHER_CKPT_DIR),
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

with open(RESULTS / "teacher_ppo_config.json", "w") as f:
    json.dump(dict(
        env_name=ENV_NAME,
        seed=TRAIN_SEED,
        checkpoint_dir=str(TEACHER_CKPT_DIR),
        ppo_params=params_cfg.to_dict(),
    ), f, indent=2, default=str)

with open(RESULTS / "teacher_reward_curve.json", "w") as f:
    json.dump(dict(
        steps=x_data,
        reward_mean=y_data,
        reward_std=y_err,
        time_to_jit_s=(times[1] - times[0]).total_seconds(),
        time_to_train_s=(times[-1] - times[1]).total_seconds(),
    ), f, indent=2)

plt.errorbar(x_data, y_data, yerr=y_err, color="blue")
plt.xlabel("# environment steps")
plt.ylabel("reward per episode")
plt.title(f"PPO teacher, {ENV_NAME}, seed {TRAIN_SEED}")
plt.savefig(RESULTS / "teacher_reward_curve.png", dpi=150, bbox_inches="tight")
print(f"saved checkpoints to {TEACHER_CKPT_DIR} and results to {RESULTS}")