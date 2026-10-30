"""Shared helpers for visualize_demos.py and visualize_eval.py.

Both the demo dataset and the evaluation trajectories use the same flat
format (per-step arrays split by episode_ends, plus reset_keys,
init_box_pos, target_pos and per-episode labels), so everything here works
on either.
"""

from common import SUCCESS_THRESH, dist_to_target, env, env_cfg, save_video  # must be first

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jp

DT = float(env.dt)
EPISODE_LENGTH = env_cfg.episode_length
SAMPLE_HALF_WIDTH = 0.2   # pick.py samples cube start and target within ±0.2 m in x, y

CATEGORIES = [
    "success",
    "reached target, not held",
    "grasped, not delivered",
    "never reached cube",
    "knocked out of bounds",
]
CATEGORY_COLORS = ["tab:green", "tab:olive", "tab:orange", "tab:red", "tab:purple"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path):
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    ends = d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    d["episodes"] = list(zip(starts, ends))
    return d


def n_episodes(d):
    return len(d["episodes"])


def episode_lengths(d):
    """Per-episode lengths, computed from episode_ends (no 'length' key needed)."""
    return np.diff(np.concatenate([[0], d["episode_ends"]]))


def episode(d, key, i):
    s, e = d["episodes"][i]
    return d[key][s:e]


def meta(d, key, default=None):
    """A scalar stored as a 0-d array in the npz (save_evals stores these as
    strings, e.g. eval_seed='1000'). Returns a str, or default if missing."""
    return str(d[key]) if key in d else default


# ---------------------------------------------------------------------------
# Replay (reset with the stored key, step through the stored actions)
# ---------------------------------------------------------------------------
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


def replay(d, i):
    """Replay what was actually executed: executed_actions if present (noisy
    demos store clean labels in `actions`), otherwise actions."""
    key = "executed_actions" if "executed_actions" in d else "actions"
    state = jit_reset(jp.asarray(d["reset_keys"][i]))
    states = [state]
    for a in episode(d, key, i):
        state = jit_step(state, jp.asarray(a))
        states.append(state)
    return states


def check_replay(d, i, states):
    """Max |replayed - stored| distance; ~0 means the replay is faithful."""
    replayed = np.array([dist_to_target(s) for s in states[1:]])
    return float(np.max(np.abs(replayed - episode(d, "dists", i))))


def save_replay_video(d, i, path):
    """Replay episode i, save it as a video, return the replay error."""
    states = replay(d, i)
    save_video(states, path)
    return check_replay(d, i, states)


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------
def failure_categories(d):
    """One category name per episode (see CATEGORIES)."""
    lengths = episode_lengths(d)
    ever_reached = d["ever_reached"] if "ever_reached" in d else d["min_dist"] < SUCCESS_THRESH
    cats = []
    for i in range(n_episodes(d)):
        if d["success"][i]:
            cats.append("success")
        elif lengths[i] < EPISODE_LENGTH:
            cats.append("knocked out of bounds")
        elif ever_reached[i]:
            cats.append("reached target, not held")
        elif d["reached_box"][i]:
            cats.append("grasped, not delivered")
        else:
            cats.append("never reached cube")
    return cats


def category_counts(d):
    cats = failure_categories(d)
    return {c: cats.count(c) for c in CATEGORIES}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _sampling_region(ax):
    """Draw pick.py's ±0.2 m sampling square around the initial cube position."""
    cx, cy = np.asarray(env._init_obj_pos)[:2]
    w = SAMPLE_HALF_WIDTH
    ax.add_patch(patches.Rectangle((cx - w, cy - w), 2 * w, 2 * w, fill=False,
                                   ls="--", color="0.5", label="sampling region"))


def plot_starts(d, path, title, outcome=None, background=None, show_targets=False):
    """Top-down cube start positions.

    outcome:      optional bool array -> green (success) / red (failure)
    background:   optional second dataset whose starts are drawn faintly (e.g. demos)
    show_targets: also draw each target (colour = height) linked to its start
    """
    box, tgt = d["init_box_pos"], d["target_pos"]
    fig, ax = plt.subplots(figsize=(6, 6))
    _sampling_region(ax)

    if background is not None:
        bg = background["init_box_pos"]
        ax.scatter(bg[:, 0], bg[:, 1], s=12, color="0.75",
                   label=f"training demo starts ({len(bg)})")

    if show_targets:
        for b, t in zip(box, tgt):
            ax.plot([b[0], t[0]], [b[1], t[1]], color="0.85", lw=0.6, zorder=0)
        sc = ax.scatter(tgt[:, 0], tgt[:, 1], s=18, marker="x", c=tgt[:, 2],
                        cmap="autumn", label="target (colour = height)")
        fig.colorbar(sc, ax=ax, label="target z (m)", shrink=0.8)

    if outcome is None:
        ax.scatter(box[:, 0], box[:, 1], s=18, color="tab:blue", label="cube start")
    else:
        outcome = np.asarray(outcome, dtype=bool)
        ax.scatter(box[outcome, 0], box[outcome, 1], s=30, color="tab:green",
                   label=f"success ({outcome.sum()})")
        ax.scatter(box[~outcome, 0], box[~outcome, 1], s=30, color="tab:red", marker="x",
                   label=f"failure ({(~outcome).sum()})")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _padded(curves):
    """Stack curves of different lengths, padding with NaN."""
    n = max(len(c) for c in curves)
    out = np.full((len(curves), n), np.nan)
    for i, c in enumerate(curves):
        out[i, :len(c)] = c
    return out


def plot_dist_curves(groups, path, title):
    """groups: list of (dataset, label, color). Thin line per episode + median."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, label, color in groups:
        curves = [episode(d, "dists", i) for i in range(n_episodes(d))]
        for c in curves:
            ax.plot(np.arange(1, len(c) + 1) * DT, c, color=color, alpha=0.15, lw=0.8)
        med = np.nanmedian(_padded(curves), axis=0)
        ax.plot(np.arange(1, len(med) + 1) * DT, med, color=color, lw=2,
                label=f"{label} (median)")
    ax.axhline(SUCCESS_THRESH, color="black", ls="--", lw=1,
               label=f"{SUCCESS_THRESH} m threshold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cube-target distance (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)