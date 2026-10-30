"""Compare teacher and student evaluations across dataset sizes (Part 4).

Usage (from the project root), after eval_teacher.py and eval_student.py:
    uv run python code/visualize_eval.py                       # sizes 25 50 100 200 500, eval seed 1000
    uv run python code/visualize_eval.py --eval-seed 2000
    uv run python code/visualize_eval.py --sizes 25 100 500
    uv run python code/visualize_eval.py --prefixes student_demos100_es1000 ...   # explicit

For each size N it reads results/student_demos<N>_es<seed>_eval_traj.npz, plus
the teacher's evaluation on the same seed (results/teacher_es<seed>_eval_traj.npz,
or results/teacher_eval_traj.npz). All policies must have been evaluated on the
same starts -- this is checked. Missing sizes are skipped with a message.

The action horizon (for the replan marks in the action plots) is read from the
evaluation file if it was recorded there, otherwise from the config saved in the
student's checkpoint.

Outputs (results/eval_es<seed>/ and videos/<prefix>/):
  failure_modes.png                       failure-category counts, teacher and every size
  success_map_<prefix>.png                eval starts coloured by outcome, demo starts faint
  dist_curves_<prefix>.png                teacher vs student distance curves
  actions_<prefix>_<category>_ep<i>.png   teacher vs student actions, one start per
                                          failure category (student fails, teacher succeeds)
  replay_ep<i>_<category>.mp4             replay of the same episode as each action plot,
                                          plus replay_ep<i>_success.mp4
  eval_analysis.json                      category counts, teacher/student agreement,
                                          failure ids, chosen example episodes
"""

from common import RESULTS, ROOT, VIDEOS  # must be first
from viz_utils import (CATEGORIES, CATEGORY_COLORS, DT, category_counts, episode,
                       failure_categories, load_dataset, meta, plot_dist_curves,
                       plot_starts, save_replay_video)

import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SIZES = (25, 50, 100, 200, 500)


def slugify(category):
    return category.replace(",", "").replace(" ", "_")


def label_for(prefix):
    """'student_demos100_es1000' -> '100 demos'; anything else unchanged."""
    m = re.match(r"student_demos(\d+)", prefix)
    return f"{m.group(1)} demos" if m else prefix


def action_horizon_of(d):
    """Action horizon of an evaluation: from the eval file if recorded, else from
    the training config stored in the checkpoint it names. None if neither works."""
    ah = meta(d, "action_horizon")
    if ah not in (None, "None"):
        return int(ah)
    path = meta(d, "policy")
    if path is None:
        return None
    try:
        import torch
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        return int(cfg["data"]["action_horizon"])
    except Exception as e:                      # missing checkpoint etc.: plot without marks
        print(f"could not read the action horizon from {path}: {e}")
        return None


def plot_failure_modes(policies, path):
    names = list(policies)
    counts = {p: category_counts(d) for p, d in policies.items()}
    x = np.arange(len(names))
    width = 0.8 / len(CATEGORIES)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(names)), 4))
    for j, (cat, color) in enumerate(zip(CATEGORIES, CATEGORY_COLORS)):
        vals = [counts[p][cat] for p in names]
        bars = ax.bar(x + (j - (len(CATEGORIES) - 1) / 2) * width, vals, width,
                      label=cat, color=color)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(p) for p in names])
    ax.set_ylabel("episodes")
    ax.set_title("Outcome of each evaluation episode")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts


def plot_action_compare(teacher, student, i, label, action_horizon, path):
    ta, sa = episode(teacher, "actions", i), episode(student, "actions", i)
    n_dim = ta.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(8, 1.4 * n_dim), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(np.arange(len(ta)) * DT, ta[:, j], color="tab:blue", label="teacher")
        ax.plot(np.arange(len(sa)) * DT, sa[:, j], color="tab:orange", label=label)
        if action_horizon:
            for t in range(0, len(sa), action_horizon):   # replanning points
                ax.axvline(t * DT, color="0.85", lw=0.5, zorder=0)
        ax.set_ylabel(f"a[{j}]")
        ax.set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=8, loc="upper right")
    xlabel = "time (s)" + ("  (grey lines: student replans)" if action_horizon else "")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"Actions on eval start {i}: teacher vs {label}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_per_category(cats, s_ok, t_ok):
    """First episode of each failure category where the student fails and the
    teacher succeeds on the same start: {category: episode}."""
    contrast = np.flatnonzero(~s_ok & t_ok)
    examples = {}
    for cat in CATEGORIES[1:]:                         # skip "success"
        eps = [int(i) for i in contrast if cats[i] == cat]
        if eps:
            examples[cat] = eps[0]
    return examples


def find_teacher(name, seed):
    """Teacher eval file: the given prefix, else teacher_es<seed>, else teacher."""
    candidates = [name] if name else [f"teacher_es{seed}", "teacher"]
    for c in candidates:
        if (RESULTS / f"{c}_eval_traj.npz").exists():
            return c
    raise SystemExit(f"no teacher evaluation found (tried {candidates}); "
                     f"run eval_teacher.py or pass --teacher")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help="dataset sizes to compare (conditions demos<N>)")
    ap.add_argument("--eval-seed", type=int, default=1000)
    ap.add_argument("--prefixes", nargs="+", default=None,
                    help="explicit eval prefixes instead of --sizes")
    ap.add_argument("--teacher", default=None,
                    help="teacher eval prefix (default: teacher_es<seed>, then teacher)")
    ap.add_argument("--demos", default="data/demos_pool_1000.npz",
                    help="demo dataset drawn faintly on the success maps")
    ap.add_argument("--out", default=None,
                    help="subfolder of results/ for the figures (default: eval_es<seed>)")
    args = ap.parse_args()

    if args.prefixes:
        wanted = args.prefixes
    else:
        wanted = [f"student_demos{n}_es{args.eval_seed}" for n in args.sizes]
    prefixes = []
    for p in wanted:
        if (RESULTS / f"{p}_eval_traj.npz").exists():
            prefixes.append(p)
        else:
            print(f"missing results/{p}_eval_traj.npz, skipped")
    if not prefixes:
        raise SystemExit("no student evaluations found")

    OUT = RESULTS / (args.out or f"eval_es{args.eval_seed}")
    OUT.mkdir(parents=True, exist_ok=True)

    demos = load_dataset(ROOT / args.demos)
    teacher_prefix = find_teacher(args.teacher, args.eval_seed)
    teacher = load_dataset(RESULTS / f"{teacher_prefix}_eval_traj.npz")
    students = {p: load_dataset(RESULTS / f"{p}_eval_traj.npz") for p in prefixes}
    policies = {"teacher": teacher, **students}

    for p, d in students.items():
        assert np.array_equal(d["reset_keys"], teacher["reset_keys"]), \
            f"{p} was not evaluated on the same starts as {teacher_prefix}"

    counts = plot_failure_modes(policies, OUT / "failure_modes.png")
    plot_starts(teacher, OUT / "success_map_teacher.png", "Teacher: outcome per eval start",
                outcome=teacher["success"], background=demos)

    analysis = dict(n_episodes=len(teacher["success"]),
                    eval_seed=meta(teacher, "eval_seed"),
                    teacher=teacher_prefix,
                    category_counts={label_for(p): c for p, c in counts.items()},
                    success_rate={label_for(p): float(d["success"].mean())
                                  for p, d in policies.items()},
                    students={})

    for p, s in students.items():
        name = label_for(p)
        plot_starts(s, OUT / f"success_map_{p}.png", f"{name}: outcome per eval start",
                    outcome=s["success"], background=demos)
        plot_dist_curves([(teacher, "teacher", "tab:blue"), (s, name, "tab:orange")],
                         OUT / f"dist_curves_{p}.png", f"Distance to target: teacher vs {name}")

        t_ok, s_ok = teacher["success"].astype(bool), s["success"].astype(bool)
        cats = failure_categories(s)
        failures = np.flatnonzero(~s_ok)
        examples = example_per_category(cats, s_ok, t_ok)

        # one action plot + one replay video per failure category, same episode
        hz = action_horizon_of(s)
        replay_err = {}
        for cat, i in examples.items():
            slug = slugify(cat)
            plot_action_compare(teacher, s, i, f"{name} ({cat})", hz,
                                OUT / f"actions_{p}_{slug}_ep{i}.png")
            vid = f"replay_ep{i}_{slug}.mp4"
            replay_err[vid] = save_replay_video(s, i, VIDEOS / p / vid)

        # plus one success for comparison
        if s_ok.any():
            i = int(np.flatnonzero(s_ok)[0])
            vid = f"replay_ep{i}_success.mp4"
            replay_err[vid] = save_replay_video(s, i, VIDEOS / p / vid)

        analysis["students"][name] = dict(
            prefix=p,
            policy=meta(s, "policy"),
            action_horizon=hz,
            both_succeed=int((s_ok & t_ok).sum()),
            student_fails_teacher_succeeds=int((~s_ok & t_ok).sum()),
            teacher_fails_student_succeeds=int((s_ok & ~t_ok).sum()),
            both_fail=int((~s_ok & ~t_ok).sum()),
            failures={int(i): cats[i] for i in failures},
            examples_per_category=examples,
            replay_videos=replay_err,
        )

    with open(OUT / "eval_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(json.dumps(analysis, indent=2))
    print(f"figures in {OUT}")