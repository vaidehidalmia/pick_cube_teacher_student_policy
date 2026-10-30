
from common import TEACHER_CKPT_DIR, latest_checkpoint, load_teacher
import jax
import torch
import collections
import numpy as np
from diffusion_model import ConditionalUnet1D
from diffusers import DDPMScheduler

from student_dataset import normalize_data, unnormalize_data

class TeacherPolicy():
    def __init__(self, path=TEACHER_CKPT_DIR):
        self.teacher_path = latest_checkpoint(path)
        print("loading teacher from", self.teacher_path)
        self.teacher = load_teacher(self.teacher_path)
        self.key = jax.random.PRNGKey(0)
    
    def reset(self):
        return

    def act(self, obs):
        action, _ = self.teacher(obs, self.key)
        return action


class DiffusionPolicy():
    def __init__(self, path, device, seed):
        self.device = device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        
        self.cfg = ckpt["config"]
        self.stats = ckpt["stats"]
        self.obs_dim = ckpt["obs_dim"]
        self.action_dim = ckpt["action_dim"]
        self.obs_horizon = self.cfg["data"]["obs_horizon"]
        self.pred_horizon = self.cfg["data"]["pred_horizon"]
        self.action_horizon = self.cfg["data"]["action_horizon"]
        self.num_diffusion_iters = self.cfg["diffusion"]["num_train_timesteps"]
        
        
        self.student_policy = ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=self.obs_dim * self.obs_horizon,
                diffusion_step_embed_dim=self.cfg["model"]["diffusion_step_embed_dim"],
                down_dims=self.cfg["model"]["down_dims"],
                kernel_size=self.cfg["model"]["kernel_size"],
                n_groups=self.cfg["model"]["n_groups"],
            )
        self.student_policy.load_state_dict(ckpt["model"])
        self.student_policy.to(device)
        self.student_policy.eval()

        self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule=self.cfg["diffusion"]["beta_schedule"],     
                clip_sample=self.cfg["diffusion"]["clip_sample"],       
                prediction_type=self.cfg["diffusion"]["prediction_type"],
            )
        torch.manual_seed(seed)
        self.reset()


    
    def reset(self):
        self.obs_deque = None
        self.action_queue = collections.deque()
        
    def act(self, obs):
        obs = np.asarray(obs)
        if self.obs_deque is None:
            self.obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)
        else:
            self.obs_deque.append(obs)

        if not self.action_queue:
            self._plan()
            
        return self.action_queue.popleft().clip(-1, 1)

    def _plan(self):
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(self.obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=self.stats['obs'])
        
        # device transfer
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon , self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.student_policy(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats['action'])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        actions = action_pred[start:end,:]
        # (action_horizon, action_dim)
        self.action_queue.extend(actions) 




    