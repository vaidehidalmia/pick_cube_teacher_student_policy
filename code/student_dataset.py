#Source: https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=93E9RdnR4D8v

import numpy as np
import torch


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def _range(stats, eps=1e-6):
    """max - min, guarded so constant dimensions don't divide by zero."""
    return np.maximum(stats["max"] - stats["min"], eps)
 
 
def normalize_data(data, stats):
    ndata = (data - stats["min"]) / _range(stats)   # -> [0, 1]
    return ndata * 2 - 1                            # -> [-1, 1]
 
 
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * _range(stats) + stats["min"]



def select_episodes(obs, actions, episode_ends, episode_ids):
    """Return obs, actions and a rebuilt episode_ends for just these episodes."""
    new_obs, new_actions, lengths = [], [], []
    for i in episode_ids:
        start = 0 if i == 0 else episode_ends[i - 1]
        end = episode_ends[i]
        ep_obs = obs[start:end]
        new_obs.append(ep_obs)
        new_actions.append(actions[start:end])
        lengths.append(len(ep_obs))
    return np.concatenate(new_obs), np.concatenate(new_actions), np.cumsum(lengths)


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, episode_ids, pred_horizon, obs_horizon, action_horizon):
        with np.load(npz_path) as f:
            obs = f["obs"]
            actions = f["actions"]
            episode_ends = f["episode_ends"]
            n_ep = len(episode_ends)

        obs, actions, episode_ends = select_episodes(obs, actions, episode_ends, episode_ids)
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, obs_dim)
            "obs": obs.astype(np.float32), 
            # (N, action_dim)         
            "action": actions.astype(np.float32),   
        }

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # # compute statistics for selected episodes and normalized data to [-1,1]
        self.stats = {}
        self.normalized_train_data = {}
        for key, arr in train_data.items():
            self.stats[key] = get_data_stats(arr)
            self.normalized_train_data[key] = normalize_data(arr, self.stats[key]).astype(np.float32)
 
        self.episode_ids = np.asarray(episode_ids)
        self.episode_ends = episode_ends
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample


if __name__ == "__main__":
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "demos_baseline_100.npz"
 
    # toy test for select_episodes
    o, a, e = select_episodes(np.arange(12), np.arange(12), np.array([4, 7, 12]), [0, 2])
    print(o, e)                                       # [0 1 2 3 7 8 9 10 11] [4 9]
 
    ds = DemoDataset(path, episode_ids=range(100),
                     pred_horizon=16, obs_horizon=2, action_horizon=8)
    print(len(ds))                                    # 14300
    s = ds[0]
    print(s["obs"].shape, s["action"].shape)          # (2, 66) (16, 8)
    print(s["obs"].min(), s["obs"].max())             # within [-1, 1]
    print(np.array_equal(s["obs"][0], s["obs"][1]))   # True: first window pads frame 0
 
    smallest = {k: float(_range(v).min()) for k, v in ds.stats.items()}
    print("smallest range per key:", smallest)       # watch for ~0 values
 
    x = ds.normalized_train_data["action"]
    print(np.abs(normalize_data(unnormalize_data(x, ds.stats["action"]),
                                ds.stats["action"]) - x).max())   # ~1e-6
 
    print(len(DemoDataset(path, range(25), 16, 2, 8)))  # 3575