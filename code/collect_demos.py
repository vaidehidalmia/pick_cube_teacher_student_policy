from common import (CFG, RESULTS, ROOT, SUCCESS_THRESH, TEACHER_CKPT_DIR, VIDEOS, save_video)
from policies import TeacherPolicy
from rollouts import run_episode, summarize

import json
import jax
import numpy as np

def collect_pool(teacher, cfg):
    keys = jax.random.split(jax.random.PRNGKey(cfg["demos_seed"]), cfg["n_episodes"])
    demos, videos = [], []
    for i, key in enumerate(np.asarray(keys)):
        ep, states = run_episode(teacher, key, SUCCESS_THRESH, record=i < cfg["n_videos"])
        ep["episode"] = i
        demos.append(ep)
        if states:
            videos.append(states)
        print(f"demo {i:4d}: success={ep['success']} final_dist={ep['final_dist']:.4f}")
    return demos, videos


def save_dataset(demos, path):
    np.savez_compressed(
        path,
        # per step: concatenated, split by episode_ends
        obs=np.concatenate([d["obs"] for d in demos]),
        actions=np.concatenate([d["actions"] for d in demos]),
        rewards=np.concatenate([d["rewards"] for d in demos]),
        dists=np.concatenate([d["dists"] for d in demos]),
        episode_ends=np.cumsum([d["length"] for d in demos]),
        # per episode
        reset_keys=np.stack([d["reset_key"] for d in demos]),
        init_box_pos=np.stack([d["init_box_pos"] for d in demos]),
        target_pos=np.stack([d["target_pos"] for d in demos]),
        success=np.array([d["success"] for d in demos]),
        ever_reached=np.array([d["ever_reached"] for d in demos]),
        reached_box=np.array([d["reached_box"] for d in demos]),
        final_dist=np.array([d["final_dist"] for d in demos], dtype=np.float32),
        min_dist=np.array([d["min_dist"] for d in demos], dtype=np.float32),
        first_reached_step=np.array([d["first_reached_step"] for d in demos]),
        final_success_start_step=np.array([d["final_success_start_step"] for d in demos]),
        length=np.array([d["length"] for d in demos]),
    )
    with np.load(path) as d:                                     # quick check
        assert d["episode_ends"][-1] == len(d["obs"]) == len(d["actions"])
        assert np.all(np.abs(d["actions"]) <= 1.0 + 1e-6)
    print(f"saved {path}: {len(demos)} demos")


if __name__ == "__main__":                            
    cfg = CFG[f"demos_pool"]
    out = ROOT / cfg["out"]
    if out.exists():
        raise SystemExit(f"{out} already exists; move it away to recollect")
    out.parent.mkdir(parents=True, exist_ok=True)

    teacher = TeacherPolicy(TEACHER_CKPT_DIR)
    demos, videos = collect_pool(teacher, cfg)
    save_dataset(demos, out)

    stats = dict(config=cfg, teacher=str(teacher.teacher_path),
                 **summarize(demos),
                 n_successful=int(sum(d["success"] for d in demos))
                )
    for k, v in stats.items():
        print(f"{k}: {v}")
    with open(RESULTS / f"dataset_stats_pool.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    for i, states in enumerate(videos):
        save_video(states, VIDEOS / f"demos_pool" / f"demo_{i}.mp4")