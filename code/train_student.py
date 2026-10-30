#Source: https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=93E9RdnR4D8v

from common import RESULTS, ROOT  

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from diffusion_model import ConditionalUnet1D
from student_dataset import DemoDataset



def load_config(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_episodes(cfg, npz_path):
    """Return the sorted list of episode ids for this condition.

    data.episodes:
      all                 - every episode in the file
      all_successful      - only episodes labelled successful
      first_n_successful  - the first n_demos successful episodes; nested
                            (the 25-demo set is inside the 50-demo set, ...)
      random_subset       - n_demos episodes drawn (without replacement) from the
                            successful ones, using data.subset_seed
    """
    data_cfg = cfg["data"]
    mode = data_cfg["episodes"]

    with np.load(npz_path) as f:
        success = f["success"]

    if mode == "all":
        ids = np.arange(len(success))
    elif mode == "all_successful":
        ids = np.flatnonzero(success)
    elif mode == "first_n_successful":
        ids = np.flatnonzero(success)[:data_cfg["n_demos"]]
        if len(ids) < data_cfg["n_demos"]:
            raise ValueError(f"asked for {data_cfg['n_demos']} demos but only "
                             f"{len(ids)} are successful")
    elif mode == "random_subset":
        pool = np.flatnonzero(success)
        n = data_cfg["n_demos"]
        if n > len(pool):
            raise ValueError(f"asked for {n} demos but only {len(pool)} are successful")
        rng = np.random.default_rng(data_cfg["subset_seed"])
        ids = np.sort(rng.choice(pool, size=n, replace=False))
    else:
        raise ValueError(f"unknown data.episodes mode: {mode!r}")

    print(f"using {len(ids)} episodes ({mode})")
    return ids.tolist()


def save_loss(losses, num_steps, num_epochs, tag):
    """Per-step loss to JSON, then a smoothed log-scale plot."""
    with open(RESULTS / f"student_{tag}_loss.json", "w") as f:
        json.dump(dict(loss=losses, num_steps=num_steps, num_epochs=num_epochs), f)

    w = min(100, len(losses))
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    plt.figure(figsize=(7, 4))
    plt.plot(losses, alpha=0.3, label="per step")
    plt.plot(np.arange(w - 1, len(losses)), smooth, label=f"moving avg ({w})")
    plt.yscale("log")
    plt.xlabel("gradient step")
    plt.ylabel("noise-prediction MSE")
    plt.title(f"Diffusion Policy training loss ({tag})")
    plt.legend()
    plt.savefig(RESULTS / f"student_{tag}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


def main(cfg_path):
    cfg = load_config(cfg_path)
    tag = cfg["condition"]
    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda")

    data_cfg, model_cfg = cfg["data"], cfg["model"]
    diffusion_cfg, training_cfg = cfg["diffusion"], cfg["training"]
    obs_horizon = data_cfg["obs_horizon"]
    
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=f"student_{tag}", config=cfg)

    npz_path = ROOT / data_cfg["path"]
    episode_ids = choose_episodes(cfg, npz_path)
    dataset = DemoDataset(
        npz_path, episode_ids,
        pred_horizon=data_cfg["pred_horizon"],
        obs_horizon=obs_horizon,
        action_horizon=data_cfg["action_horizon"],
    )

    num_workers = training_cfg["num_workers"]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),   # reproducible shuffling
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    obs_dim = dataset.normalized_train_data["obs"].shape[-1]
    action_dim = dataset.normalized_train_data["action"].shape[-1]
    print(f"condition: {tag}, {len(dataset)} windows, obs_dim {obs_dim}, action_dim {action_dim}")

    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        diffusion_step_embed_dim=model_cfg["diffusion_step_embed_dim"],
        down_dims=model_cfg["down_dims"],
        kernel_size=model_cfg["kernel_size"],
        n_groups=model_cfg["n_groups"],
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_cfg["num_train_timesteps"],
        beta_schedule=diffusion_cfg["beta_schedule"],     
        clip_sample=diffusion_cfg["clip_sample"],        
        prediction_type=diffusion_cfg["prediction_type"], 
    )

    
    num_steps = training_cfg["num_steps"]               
    num_epochs = int(np.ceil(num_steps / len(dataloader)))
    print(f"training for {num_steps} steps ({num_epochs} epochs of {len(dataloader)} batches)")

    
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=noise_pred_net.parameters(), power=training_cfg["ema_power"])

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=training_cfg["lr"], weight_decay=training_cfg["weight_decay"])
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=training_cfg["warmup_steps"],
        num_training_steps=num_steps,
    )

    
    losses = []
    with tqdm(range(num_epochs), desc="Epoch") as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = []
            # batch loop
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    if len(losses) >= num_steps:      # exact step budget
                        break
                    nobs = nbatch["obs"].to(device)            # (B, obs_horizon, obs_dim)
                    naction = nbatch["action"].to(device)      # (B, pred_horizon, action_dim)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :obs_horizon, :]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)
                    
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device).long()
                    
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()                        
                    
                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    losses.append(loss_cpu)
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    if use_wandb:
                        wandb.log({"train/loss": loss_cpu,
                                   "train/lr": lr_scheduler.get_last_lr()[0],
                                   "train/epoch": epoch_idx},
                                  step=len(losses))
            if epoch_loss:
                tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema.copy_to(noise_pred_net.parameters())

    # Save everything evaluation needs
    torch.save({
        "model": noise_pred_net.state_dict(),
        "stats": dataset.stats,
        "config": cfg,
        "episode_ids": dataset.episode_ids,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_steps": num_steps,
    }, ckpt_dir / "policy.pt")
    save_loss(losses, num_steps, num_epochs, tag)
    print(f"saved {ckpt_dir / 'policy.pt'} ({num_steps} steps) and "
          f"results/student_{tag}_loss.{{json,png}}")

    if use_wandb:
        wandb.summary["num_steps"] = num_steps
        wandb.summary["final_loss_avg100"] = float(np.mean(losses[-100:]))
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1])