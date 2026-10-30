from common import CFG, SUCCESS_THRESH, TEACHER_CKPT_DIR
from evaluation import evaluate, save_evals
from policies import TeacherPolicy

import argparse
 
DEFAULT_EVAL_SEED = CFG["eval"]["seed"]
N_EPISODES = CFG["eval"]["n_episodes"]
N_VIDEOS = CFG["eval"]["n_videos"]
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval-seed", type=int, default=None)
    args = p.parse_args()

    eval_seed = (
        DEFAULT_EVAL_SEED
        if args.eval_seed is None
        else args.eval_seed
    )
    teacher = TeacherPolicy(TEACHER_CKPT_DIR)
    prefix = f"teacher_es{eval_seed}"

    episodes, rollouts = evaluate(
            teacher,
            N_EPISODES,
            eval_seed,
            SUCCESS_THRESH,
            n_record=N_VIDEOS,
        )
    
    save_evals(
        prefix,
        episodes,
        rollouts,
        teacher.teacher_path,
        eval_seed,
        SUCCESS_THRESH
    )