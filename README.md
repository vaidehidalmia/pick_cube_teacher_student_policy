# Imitation Learning on PandaPickCube

A PPO teacher collects demonstrations in MuJoCo Playground's `PandaPickCube`
task, and a state-based **Diffusion Policy** student is trained on them. The
experiment varies **how many demonstrations** the student is trained on
(25, 50, 100, 200, 500) with everything else held fixed, and evaluates each
student on the same 150 held-out starts.

The task: pick up the cube and bring it to a target 20–40 cm above the table,
then hold it there until the episode ends (150 steps, 3 s). An episode counts
as a success if the cube is within **2 cm** of the target at the last step.

## Results

150 evaluation episodes per row (eval seeds 1000, 2000, 3000 × 50), all rows on
the same starts.

| Demos | Success (2 cm) | Reached cube | Ever reached target | Median final dist |
|---|---|---|---|---|
| 25 | 0.0% | 29.3% | 1.3% | 33.9 cm |
| 50 | 5.3%  | 49.3% | 13.3% | 17.6 cm |
| 100 (baseline) | 18.7%  | 72.0% | 26.7% | 4.6 cm |
| 200 | 56.0%  | 88.0% | 60.7% | 1.7 cm |
| 500 | 72.0% | 93.3% | 78.0% | 1.1 cm |
| PPO teacher | 95.3%  | 98.7% | 95.3% | 0.48 cm |


## Repository layout

```
report.pdf
README.md
pyproject.toml, uv.lock       exact Python dependencies
configs/
  config.yaml                 environment, teacher, demo collection and eval settings
  student_demos<N>.yaml       one config per dataset size (only n_demos differs)
code/
  common.py                   environment setup, teacher loading, shared helpers
  train_teacher.py            PPO teacher (MuJoCo Playground defaults)
  collect_demos.py            runs the teacher, saves the demo pool
  eval_teacher.py             evaluates the teacher on one eval seed
  student_dataset.py          windowed dataset + min-max normalization
  diffusion_model.py          conditional 1D U-Net (from the Diffusion Policy Colab)
  train_student.py            Diffusion Policy training
  policies.py                 teacher and student policy wrappers
  rollouts.py, evaluation.py  episode rollouts, metrics, saving results
  eval_student.py             evaluates one student on one eval seed
  run_all.sh                  trains and evaluates every dataset size
  visualize_demos.py          demo coverage, consistency, failures
  visualize_eval.py           per-seed comparison of teacher and students
  plot_vs_size.py             figures across dataset sizes
data/demos_pool_1000.npz      demo pool (1,050 episodes, 953 successful)
checkpoints/teacher/          PPO teacher checkpoint
results/                      JSON summaries and figures
videos/                       demo, teacher and student rollout videos
```

Student checkpoints are not included; `run_all.sh` rebuilds them.

## Setup

```bash
git clone https://github.com/vaidehidalmia/pick_cube_teacher_student_policy.git
cd pick_cube_teacher_student_policy
uv sync
source .venv/bin/activate
```

All commands are run from the repository root.

## Reproducing the results

The teacher checkpoint and the demo pool are included, so steps 1–2 are optional.

```bash
# 1. (optional) train the PPO teacher -> checkpoints/teacher/
python code/train_teacher.py

# 2. (optional) collect the demo pool -> data/demos_pool_1000.npz
python code/collect_demos.py

# 3. evaluate the teacher on the three eval seeds
for s in 1000 2000 3000; do python code/eval_teacher.py --eval-seed $s; done

# 4. train and evaluate every student (25/50/100/200/500 demos, 3 eval seeds each)
bash code/run_all.sh          # skips anything already done; safe to re-run

# 5. figures
python code/visualize_demos.py                       
python code/plot_vs_size.py         
```

## Experimental setup

- **Teacher:** Brax PPO with MuJoCo Playground's default `PandaPickCube`
  hyperparameters, `impl: jax`, deterministic actions when collecting demos and
  evaluating.
- **Demos:** 1,050 episodes from the teacher (90.8% successful). Each student is
  trained on the **first N successful** demos, so smaller sets are subsets of
  larger ones.
- **Student:** Diffusion Policy (DDPM, 100 diffusion steps, squared-cosine
  schedule, noise prediction with MSE). Observation horizon 2, prediction horizon
  16, action horizon 4. Conditional 1D U-Net with dims [256, 512, 1024], FiLM
  conditioning on the observations. Observations and actions are min-max
  normalized to [-1, 1] with each training subset's statistics.
- **Training:** AdamW, lr 1e-4, weight decay 1e-6, batch 256, cosine schedule
  with 500 warmup steps, **30,000 gradient steps for every dataset size**, EMA
  weights used for evaluation.
- **Actions:** 8-D in [-1, 1] (7 arm joints + gripper). The environment
  multiplies each by 0.04 and adds it to the previous joint position targets.
- **Evaluation:** 50 episodes on each of eval seeds 1000, 2000, 3000 for every
  student and the teacher; success = cube within 2 cm of the target at step 150.

### Seeds

| What | Seed |
|---|---|
| Teacher training | 1 |
| Demo collection | 0 |
| Student training | 0 |
| Evaluation | 1000, 2000, 3000 |

### Output naming

- Student evaluation: `results/student_demos<N>_es<seed>_eval.json` (summary)
  and `_eval_traj.npz` (per-step data), videos in `videos/student_demos<N>_es<seed>/`.
- Teacher evaluation: `results/teacher_es<seed>_eval.json` / `_eval_traj.npz`.
- Training loss: `results/student_demos<N>_loss.json` / `.png`.
- Logs from `run_all.sh`: `logs/`.

## References

- C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion*, RSS 2023, and its state-based Push-T Colab (source of the dataset
  and U-Net code).
- K. Zakka et al., *MuJoCo Playground*, 2025 — environment and PPO
  training setup (manipulation notebook).
- Brax PPO implementation.
