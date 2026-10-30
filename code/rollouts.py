from common import box_pos_from_qpos, dist_to_target, env, env_cfg, success_times

import jax
import numpy as np
from jax import numpy as jp

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


def run_episode(policy, reset_key, thresh, record=False):
    state = jit_reset(jp.asarray(reset_key))
    policy.reset()

    init_box_pos = np.asarray(box_pos_from_qpos(state))   # xpos is 0 at reset
    target_pos = np.asarray(state.info["target_pos"])
    states = [state] if record else None
    obs, actions, rewards, dists = [], [], [], []

    for _ in range(env_cfg.episode_length):
        obs.append(np.asarray(state.obs))
        action = np.asarray(policy.act(state.obs), dtype=np.float32)
        actions.append(action)
        state = jit_step(state, jp.asarray(action))
        rewards.append(float(state.reward))
        dists.append(dist_to_target(state))
        if record:
            states.append(state)
        if state.done:
            break

    dists = np.array(dists, dtype=np.float32)
    first_reached_step, final_success_start_step = success_times(dists, thresh)
    episode = dict(
        obs=np.stack(obs).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dists=dists,
        reset_key=np.asarray(reset_key),
        init_box_pos=init_box_pos,
        target_pos=target_pos,
        final_dist=float(dists[-1]),
        min_dist=float(dists.min()),
        success=bool(dists[-1] < thresh),
        ever_reached=bool(dists.min() < thresh),
        reached_box=bool(state.info["reached_box"]),
        first_reached_step=first_reached_step,
        final_success_start_step=final_success_start_step,
        length=len(actions),
    )
    return episode, states


def run_episodes(policy, reset_keys, thresh, n_record=0):
    episodes, rollouts = [], []
    for i, key in enumerate(reset_keys):
        ep, states = run_episode(policy, key, thresh, record=i < n_record)
        ep["episode"] = i
        episodes.append(ep)
        if states is not None:
            rollouts.append(states)
        print(f"episode {i:4d}: success={ep['success']} "
              f"final_dist={ep['final_dist']:.4f} length={ep['length']}")
    return episodes, rollouts


def summarize(episodes):
    final = np.array([e["final_dist"] for e in episodes])
    tts = np.array([e["final_success_start_step"] for e in episodes if e["success"]])
    rate = lambda k: float(np.mean([e[k] for e in episodes]))
    dt = float(env.dt)
    return dict(
        n_episodes=len(episodes),
        success_rate=rate("success"),
        ever_reached_rate=rate("ever_reached"),
        reached_box_rate=rate("reached_box"),
        final_dist_median=float(np.median(final)),
        final_dist_max=float(np.max(final)),
        final_success_start_s_mean=float(tts.mean() * dt) if len(tts) else None,
        final_success_start_s_std=float(tts.std() * dt) if len(tts) else None,
        episode_lengths=sorted({e["length"] for e in episodes}),
    )