"""Inspect a demonstration dataset (Part 1: consistency, coverage, errors).

Usage (from anywhere):
    uv run python code/visualize_demos.py                  # the whole pool
    uv run python code/visualize_demos.py --first-n 100    # the 100-demo baseline
    uv run python code/visualize_demos.py --data data/other_demos.npz

--data is the demo file (relative to the project root). --first-n N restricts
everything to the first N successful demos (exactly the training subset of
that size, matching `episodes: first_n_successful` in the training config).

Outputs (results/<tag>/ and videos/<tag>/, tag = data file name [+ _first<N>]):
  coverage_subsets.png     (without --first-n) cube starts of the first
                           25 / 50 / 100 / 200 / 500 successful demos (the
                           training subsets of each dataset-size condition)
  coverage.png             cube starts and targets, top-down
  dist_curves.png          distance-to-target over time, all demos
  tts_vs_init_dist.png     time to success vs initial cube-target distance
  actions.png              per-dimension action mean ± std over time
  frames.png               frame strip from a few demos
  demo_analysis.json       consistency / efficiency numbers, outlier ids (among
                           successful demos) and the teacher's failed demos
  outlier_*.mp4            replayed outlier demos (successful ones)
  failure_demo<i>.mp4      replayed teacher failures (first N_FAILURE_VIDEOS only)
"""

from common import RESULTS, ROOT, VIDEOS, env  # must be first
from viz_utils import (DT, _sampling_region, episode, episode_lengths, load_dataset,
                       n_episodes, plot_dist_curves, plot_starts, replay,
                       save_replay_video)

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/demos_pool_1000.npz",
                help="demo dataset, relative to the project root")
ap.add_argument("--first-n", type=int, default=None)
ARGS = ap.parse_args()

DATA_PATH = ROOT / ARGS.data
TAG = Path(ARGS.data).stem + (f"_first{ARGS.first_n}" if ARGS.first_n else "")
FIG_DIR = RESULTS / TAG
VID_DIR = VIDEOS / TAG
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_SIZES = (25, 50, 100, 200, 500)        # panels of coverage_subsets.png
N_FAILURE_VIDEOS = 5                            # teacher failures replayed as videos


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------
def first_n_successful(d, n):
    """The first n successful episodes, as a dataset in the same format
    (exactly the training subset of size n)."""
    ids = np.flatnonzero(d["success"])[:n]
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} successful demos, asked for {n}")
    lengths = episode_lengths(d)
    n_steps, n_eps = len(d["obs"]), n_episodes(d)
    out = {}
    for k, v in d.items():
        if k == "episodes" or np.ndim(v) == 0:
            continue
        if len(v) == n_steps:                         # per-step: concatenate episodes
            out[k] = np.concatenate([episode(d, k, i) for i in ids])
        elif len(v) == n_eps:                         # per-episode: index
            out[k] = v[ids]
    out["episode_ends"] = np.cumsum(lengths[ids])
    starts = np.concatenate([[0], out["episode_ends"][:-1]])
    out["episodes"] = list(zip(starts, out["episode_ends"]))
    out["pool_index"] = ids                           # where each demo sits in the pool
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_tts_vs_init_dist(d):
    init_dist = np.linalg.norm(d["init_box_pos"] - d["target_pos"], axis=1)
    ok = d["success"].astype(bool)
    tts = d["final_success_start_step"][ok].astype(float)
    r = float(np.corrcoef(init_dist[ok], tts)[0, 1]) if ok.sum() > 2 else None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(init_dist[ok], tts * DT, s=10, alpha=0.6)
    ax.set_xlabel("initial cube-target distance (m)")
    ax.set_ylabel("time to success (s)")
    ax.set_title(f"Time to success vs task difficulty (r = {r:.2f})" if r is not None
                 else "Time to success vs task difficulty")
    fig.savefig(FIG_DIR / "tts_vs_init_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return init_dist, r


def plot_actions(d):
    # pad episodes that ended early with NaN so every episode lines up in time
    eps = [episode(d, "actions", i) for i in range(n_episodes(d))]
    T, n_dim = max(len(a) for a in eps), eps[0].shape[1]
    acts = np.full((len(eps), T, n_dim), np.nan)
    for k, a in enumerate(eps):
        acts[k, :len(a)] = a
    mean, std = np.nanmean(acts, 0), np.nanstd(acts, 0)
    t = np.arange(T) * DT
    fig, axes = plt.subplots(n_dim, 1, figsize=(7, 1.4 * n_dim), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(t, mean[:, j], color="tab:blue")
        ax.fill_between(t, mean[:, j] - std[:, j], mean[:, j] + std[:, j],
                        color="tab:blue", alpha=0.25)
        ax.set_ylabel(f"a[{j}]")
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Teacher actions across demos (mean ± std)")
    fig.savefig(FIG_DIR / "actions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frames(d, ids, steps=(0, 20, 40, 80, 149)):
    fig, axes = plt.subplots(len(ids), len(steps),
                             figsize=(2.6 * len(steps), 2.0 * len(ids)), squeeze=False)
    for r, i in enumerate(ids):
        states = replay(d, i)
        frames = env.render([states[s] for s in steps], height=240, width=320)
        for c, (s, img) in enumerate(zip(steps, frames)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {s * DT:.1f} s")
            if c == 0:
                ax.set_ylabel(f"demo {i}")
    fig.savefig(FIG_DIR / "frames.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_subsets(d, sizes=SUBSET_SIZES):
    """Cube starts of the first n successful demos, one panel per n."""
    n_succ = int(d["success"].sum())
    skipped = [n for n in sizes if n > n_succ]
    if skipped:
        print(f"coverage_subsets: skipping {skipped} (only {n_succ} successful demos)")
    sizes = [n for n in sizes if n <= n_succ]
    ncols = min(3, len(sizes))
    nrows = int(np.ceil(len(sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(sizes):
            ax.axis("off")                            # unused panel
            continue
        n = sizes[k]
        box = d["init_box_pos"][np.flatnonzero(d["success"])[:n]]
        _sampling_region(ax)
        ax.scatter(box[:, 0], box[:, 1], s=max(2, 40 / np.sqrt(n)), color="tab:blue")
        ax.set_aspect("equal")
        ax.set_title(f"{n} demos")
        if k % ncols == 0:
            ax.set_ylabel("y (m)")
        if k // ncols == nrows - 1:
            ax.set_xlabel("x (m)")
    fig.suptitle("Cube start positions covered by each training subset (top-down)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_subsets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def jerk(actions):
    return float(np.mean(np.abs(np.diff(actions, axis=0))))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = load_dataset(DATA_PATH)
    if not ARGS.first_n:
        plot_coverage_subsets(d)
    else:
        d = first_n_successful(d, ARGS.first_n)
    n = n_episodes(d)
    print(f"loaded {n} demos from {DATA_PATH}"
          + (f" (first {n} successful)" if ARGS.first_n else ""))

    plot_starts(d, FIG_DIR / "coverage.png",
                f"Initial conditions, {n} demos (top-down)", show_targets=True)
    plot_dist_curves([(d, "teacher demos", "tab:blue")], FIG_DIR / "dist_curves.png",
                     "Distance to target, all demonstrations")
    init_dist, r = plot_tts_vs_init_dist(d)
    plot_actions(d)
    plot_frames(d, ids=list(range(min(4, n))))

    # Consistency / efficiency numbers (time-to-success only defined for successes)
    succ = d["success"].astype(bool)
    tts = d["final_success_start_step"].astype(float)
    first = d["first_reached_step"].astype(float)
    jerks = np.array([jerk(episode(d, "actions", i)) for i in range(n)])
    lengths = episode_lengths(d)
    holding_frac = (lengths[succ] - tts[succ]) / lengths[succ]
    regrasp_gap = tts[succ] - first[succ]   # >0: reached target, left, came back

    # outliers among SUCCESSFUL demos only (failures are never used for training)
    succ_ids = np.flatnonzero(succ)
    worst_of = lambda x: int(succ_ids[np.argmax(x[succ])])
    outliers = dict(
        worst_final_dist=worst_of(d["final_dist"]),
        slowest=worst_of(tts),
        jerkiest=worst_of(jerks),
    )
    if np.any(regrasp_gap > 0):
        outliers["largest_reach_gap"] = int(succ_ids[np.argmax(regrasp_gap)])

    replay_err = {name: save_replay_video(d, i, VID_DIR / f"outlier_{name}_demo{i}.mp4")
                  for name, i in outliers.items()}

    # teacher failures: all listed in the JSON, only the first N_FAILURE_VIDEOS replayed
    failures = np.flatnonzero(~succ)
    recorded = set(failures[:N_FAILURE_VIDEOS].tolist())
    failure_info = {int(i): dict(final_dist=float(d["final_dist"][i]),
                                 reached_box=bool(d["reached_box"][i]),
                                 ever_reached=bool(d["ever_reached"][i]),
                                 jerk=float(jerks[i]),
                                 replay_max_dist_error=(save_replay_video(
                                     d, i, VID_DIR / f"failure_demo{i}.mp4")
                                     if i in recorded else None))
                    for i in failures}

    analysis = dict(
        dataset=str(DATA_PATH),
        subset=f"first {ARGS.first_n} successful" if ARGS.first_n else "all",
        n_demos=n,
        n_successful=int(succ.sum()),
        success_fraction=float(succ.mean()),
        episode_length=dict(mean=float(lengths.mean()), std=float(lengths.std())),
        final_success_start_s=dict(mean=float(tts[succ].mean() * DT),
                                   std=float(tts[succ].std() * DT),
                                   min=float(tts[succ].min() * DT),
                                   max=float(tts[succ].max() * DT)),
        corr_tts_vs_initial_distance=r,
        initial_box_target_dist_m=dict(mean=float(init_dist.mean()),
                                       min=float(init_dist.min()),
                                       max=float(init_dist.max())),
        init_box_pos_range_m=dict(min=d["init_box_pos"].min(0).tolist(),
                                  max=d["init_box_pos"].max(0).tolist()),
        target_pos_range_m=dict(min=d["target_pos"].min(0).tolist(),
                                max=d["target_pos"].max(0).tolist()),
        holding_fraction=dict(mean=float(holding_frac.mean()),
                              std=float(holding_frac.std())),
        demos_that_left_target_and_returned=int((regrasp_gap > 0).sum()),
        action_jerk=dict(mean=float(jerks[succ].mean()), std=float(jerks[succ].std()),
                         max=float(jerks[succ].max())),   # successful demos
        teacher_failures=failure_info,
        outliers={k: dict(demo=v,
                          final_dist=float(d["final_dist"][v]),
                          final_success_start_s=float(tts[v] * DT),
                          jerk=float(jerks[v]),
                          replay_max_dist_error=replay_err[k])
                  for k, v in outliers.items()},
    )

    with open(FIG_DIR / "demo_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(json.dumps(analysis, indent=2))
    print(f"figures in {FIG_DIR}, videos in {VID_DIR}")