from common import ENV_NAME, RESULTS, VIDEOS, save_video
from rollouts import run_episodes, summarize

import json
import jax
import numpy as np


def evaluate(policy, num_episodes, seed, thresh, n_record=0):
    reset_keys = np.asarray(jax.random.split(jax.random.PRNGKey(seed), num_episodes))
    return run_episodes(policy, reset_keys, thresh, n_record)


def save_evals(prefix, episodes, rollouts, policy_path, eval_seed, thresh):

    info = dict(env=ENV_NAME, policy=str(policy_path), eval_seed=int(eval_seed), threshold=thresh)
    
    eval_summary = {**info, **summarize(episodes)}
    with open(RESULTS / f"{prefix}_eval.json", "w") as f:
            json.dump(dict(summary=eval_summary), f, indent=2, default=str)
    np.savez_compressed(
        RESULTS / f"{prefix}_eval_traj.npz",
        # per step: concatenated, split by episode_ends
        obs=np.concatenate([e["obs"] for e in episodes]),
        actions=np.concatenate([e["actions"] for e in episodes]),
        dists=np.concatenate([e["dists"] for e in episodes]),
        episode_ends=np.cumsum([e["length"] for e in episodes]),
        # per episode
        reset_keys=np.stack([e["reset_key"] for e in episodes]),
        init_box_pos=np.stack([e["init_box_pos"] for e in episodes]),
        target_pos=np.stack([e["target_pos"] for e in episodes]),
        episode=np.array([e["episode"] for e in episodes]),
        final_dist=np.array([e["final_dist"] for e in episodes]),
        min_dist=np.array([e["min_dist"] for e in episodes]),
        success=np.array([e["success"] for e in episodes]),
        ever_reached=np.array([e["ever_reached"] for e in episodes]),
        reached_box=np.array([e["reached_box"] for e in episodes]),
        first_reached_step=np.array([e["first_reached_step"] for e in episodes]),
        final_success_start_step=np.array([e["final_success_start_step"] for e in episodes]),
        length=np.array([e["length"] for e in episodes]),
        # how the evaluation was run
        **{k: str(v) for k, v in info.items()},
    )

    for i, rollout in enumerate(rollouts):
        outcome = "success" if episodes[i]["success"] else "failure"
        save_video(rollout, VIDEOS / prefix / f"ep{i}_{outcome}.mp4")

    print(f"saved results/{prefix}_eval.json, {prefix}_eval_traj.npz "
          f"and {len(rollouts)} videos")