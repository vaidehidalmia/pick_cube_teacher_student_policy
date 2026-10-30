"""Dataset-size comparison (Part 3): one set of figures across all conditions.

Usage (from the project root), after run_all.sh and eval_teacher.py:
    uv run python code/plot_vs_size.py
    uv run python code/plot_vs_size.py --sizes 25 100 500 --eval-seeds 1000 2000 3000

For each size N it reads results/student_demos<N>_es<seed>_eval_traj.npz for
every eval seed and pools the episodes. The teacher is read from
results/teacher_es<seed>_eval_traj.npz, or from results/teacher_eval_traj.npz
when that file was evaluated on the same seed. Missing files are skipped with
a message, so this can be run while conditions are still training.

Outputs (results/vs_size/):
  success_vs_size.png        success rate vs N, 95% CI, teacher reference line
  thresholds_vs_size.png     success rate at 2 / 3 / 5 cm vs N (precision vs reach)
  failure_modes_vs_size.png  share of episodes in each outcome category, per N
  loss_curves.png            training loss of every condition (from *_loss.json)
  vs_size.json               all numbers behind the figures
If the eval files contain max_nobs (largest |normalized obs| per episode),
success_vs_size.png also shows the fraction of out-of-range episodes.
"""

from common import RESULTS  # must be first
from viz_utils import CATEGORIES, CATEGORY_COLORS, failure_categories, load_dataset, meta

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SIZES = (25, 50, 100, 200, 500)
THRESHOLDS = (0.02, 0.03, 0.05)          # metres; 0.02 is the success criterion


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n -> (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - half, centre + half


def pooled(files):
    """Pool the per-episode arrays of several eval files."""
    ds = [load_dataset(f) for f in files]
    out = dict(
        success=np.concatenate([d["success"] for d in ds]).astype(bool),
        final_dist=np.concatenate([d["final_dist"] for d in ds]).astype(float),
        categories=sum((failure_categories(d) for d in ds), []),
    )
    if all("max_nobs" in d for d in ds):
        out["max_nobs"] = np.concatenate([d["max_nobs"] for d in ds]).astype(float)
    return out


def teacher_file(seed):
    """Teacher eval file for this seed, or None."""
    f = RESULTS / f"teacher_es{seed}_eval_traj.npz"
    if f.exists():
        return f
    f = RESULTS / "teacher_eval_traj.npz"
    if f.exists() and meta(load_dataset(f), "eval_seed") == str(seed):
        return f
    return None


def summary(p):
    n = len(p["success"])
    s, lo, hi = wilson(int(p["success"].sum()), n)
    row = dict(n_episodes=n, success_rate=s, success_ci=[lo, hi],
               final_dist_median=float(np.median(p["final_dist"])),
               success_at={f"{t * 100:g}cm": float(np.mean(p["final_dist"] < t))
                           for t in THRESHOLDS},
               categories={c: p["categories"].count(c) / n for c in CATEGORIES})
    if "max_nobs" in p:
        ood = p["max_nobs"] > OOD_THRESH
        o, olo, ohi = wilson(int(ood.sum()), n)
        row.update(ood_rate=o, ood_ci=[olo, ohi])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1000, 2000, 3000])
    ap.add_argument("--ood-thresh", type=float, default=5.0)
    args = ap.parse_args()
    OOD_THRESH = args.ood_thresh

    out_dir = RESULTS / "vs_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- students -------------------------------------------------------
    rows = []
    for n in sorted(args.sizes):
        files, used = [], []
        for s in args.eval_seeds:
            f = RESULTS / f"student_demos{n}_es{s}_eval_traj.npz"
            if f.exists():
                files.append(f)
                used.append(s)
            else:
                print(f"missing {f.name}, skipped")
        if files:
            rows.append(dict(n_demos=n, eval_seeds=used, **summary(pooled(files))))
    if not rows:
        raise SystemExit("no student evaluations found")

    # ---- teacher (same seeds as the students, where available) -----------
    t_files = [f for f in (teacher_file(s) for s in args.eval_seeds) if f is not None]
    teacher = summary(pooled(t_files)) if t_files else None
    if teacher is None:
        print("no teacher evaluation on these seeds; reference lines omitted")
    elif len(t_files) < len(args.eval_seeds):
        print(f"teacher evaluated on {len(t_files)} of {len(args.eval_seeds)} seeds")

    for r in rows:
        print(f"{r['n_demos']:5d} demos  n={r['n_episodes']:4d}  "
              f"success={r['success_rate']:.2f} "
              f"[{r['success_ci'][0]:.2f}, {r['success_ci'][1]:.2f}]")

    x = np.array([r["n_demos"] for r in rows])

    def log_x(ax):
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel("number of demonstrations (log scale)")

    # ---- 1. success vs size ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.array([r["success_rate"] for r in rows])
    lo = np.array([r["success_ci"][0] for r in rows])
    hi = np.array([r["success_ci"][1] for r in rows])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=4,
                color="tab:green", label="student success")
    if all("ood_rate" in r for r in rows):
        yo = np.array([r["ood_rate"] for r in rows])
        olo = np.array([r["ood_ci"][0] for r in rows])
        ohi = np.array([r["ood_ci"][1] for r in rows])
        ax.errorbar(x, yo, yerr=[yo - olo, ohi - yo], marker="s", capsize=4,
                    color="tab:red", label=f"episodes with max |norm. obs| > {OOD_THRESH:g}")
    if teacher:
        ax.axhline(teacher["success_rate"], color="tab:blue", ls="--", lw=1,
                   label=f"teacher ({teacher['success_rate']:.2f}, "
                         f"n={teacher['n_episodes']})")
    log_x(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of evaluation episodes")
    ax.set_title(f"Success vs dataset size ({rows[0]['n_episodes']} episodes per point, 95% CI)")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "success_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. success at several distance thresholds ------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(THRESHOLDS)))
    for t, c in zip(THRESHOLDS, colors):
        key = f"{t * 100:g}cm"
        ax.plot(x, [r["success_at"][key] for r in rows], marker="o", color=c,
                label=f"student, final dist < {t * 100:g} cm")
        if teacher:
            ax.axhline(teacher["success_at"][key], color=c, ls="--", lw=0.8)
    log_x(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of evaluation episodes")
    ax.set_title("Success at looser thresholds (dashed: teacher)")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "thresholds_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. failure modes, stacked shares --------------------------------
    labels = [f"{r['n_demos']}" for r in rows] + (["teacher"] if teacher else [])
    shares = [r["categories"] for r in rows] + ([teacher["categories"]] if teacher else [])
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.2))
    bottom = np.zeros(len(labels))
    for cat, color in zip(CATEGORIES, CATEGORY_COLORS):
        vals = np.array([s[cat] for s in shares])
        ax.bar(labels, vals, bottom=bottom, color=color, label=cat)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of demonstrations")
    ax.set_ylabel("share of evaluation episodes")
    ax.set_title("Outcome of each episode, by dataset size")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_dir / "failure_modes_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. training loss of every condition ------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(args.sizes)))
    plotted = 0
    for n, c in zip(sorted(args.sizes), colors):
        f = RESULTS / f"student_demos{n}_loss.json"
        if not f.exists():
            continue
        with open(f) as fh:
            loss = np.asarray(json.load(fh)["loss"])
        w = min(200, len(loss))
        smooth = np.convolve(loss, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w - 1, len(loss)), smooth, color=c, label=f"{n} demos")
        plotted += 1
    if plotted:
        ax.set_yscale("log")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("noise-prediction MSE (moving avg 200)")
        ax.set_title("Training loss by dataset size")
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "vs_size.json", "w") as f:
        json.dump(dict(eval_seeds=args.eval_seeds, thresholds_m=list(THRESHOLDS),
                       teacher=teacher, students=rows), f, indent=2)
    print(f"figures in {out_dir}")