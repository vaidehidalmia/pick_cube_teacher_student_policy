from common import CFG, ROOT, SUCCESS_THRESH
from evaluation import evaluate, save_evals
from policies import DiffusionPolicy

import argparse

import torch
import yaml

DEFAULT_EVAL_SEED = CFG["eval"]["seed"]
N_EPISODES = CFG["eval"]["n_episodes"]
N_VIDEOS = CFG["eval"]["n_videos"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--eval-seed", type=int, default=None)
    args = p.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = ROOT / cfg["checkpoint_dir"] / "policy.pt"
    eval_seed = (
        DEFAULT_EVAL_SEED
        if args.eval_seed is None
        else args.eval_seed
    )

    student = DiffusionPolicy(
        ckpt_path,
        device=torch.device("cuda"),
        seed=cfg["seed"]
    )

    prefix = f"student_{cfg['condition']}_es{eval_seed}"

    episodes, rollouts = evaluate(
        student,
        N_EPISODES,
        eval_seed,
        SUCCESS_THRESH,
        n_record=N_VIDEOS,
    )

    save_evals(
        prefix,
        episodes,
        rollouts,
        ckpt_path,
        eval_seed,
        SUCCESS_THRESH
    )