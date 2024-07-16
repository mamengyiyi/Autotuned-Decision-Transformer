import d4rl
import gym
import numpy as np
from collections import defaultdict
# from dataset import Dataset
import time
from tqdm.auto import tqdm, trange  # noqa

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                ):
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        if filter_terminals:
            # drop terminal transitions
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

        if 'antmaze' in env_name:
            # antmaze: terminals are incorrect for GCRL
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.

            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
        else:
            dones_float = dataset['terminals'].copy()

        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        return Dataset.create(
            observations=observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
        )


def get_seq_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                ):
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    dataset['terminals'][-1] = 1
    if filter_terminals:
        # drop terminal transitions
        non_last_idx = np.nonzero(~dataset['terminals'])[0]
        last_idx = np.nonzero(dataset['terminals'])[0]
        penult_idx = last_idx - 1
        new_dataset = dict()
        for k, v in dataset.items():
            if k == 'terminals':
                v[penult_idx] = 1
            new_dataset[k] = v[non_last_idx]
        dataset = new_dataset

    if 'antmaze' in env_name:
        # antmaze: terminals are incorrect for GCRL
        dones_float = np.zeros_like(dataset['rewards'])
        dataset['terminals'][:] = 0.

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
    else:
        dones_float = dataset['terminals'].copy()

    observations = dataset['observations'].astype(obs_dtype)
    next_observations = dataset['next_observations'].astype(obs_dtype)
    # print(observations.shape)
    # exit()
    traj, traj_len = [], []
    data_, episode_step = defaultdict(list), 0
    for i in trange(observations.shape[0], desc="Processing trajectories"):
        data_["observations"].append(observations[i])
        data_["actions"].append(dataset["actions"][i])

        if 'antmaze' in env_name:
            data_["rewards"].append(dataset["rewards"][i]-1)
        else:
            data_["rewards"].append(dataset["rewards"][i])

        data_["next_observations"].append(next_observations[i])
        data_["dones_float"].append(dones_float[i])
        data_["masks"].append(1.0-dones_float[i])

        if dones_float[i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=1.0
            )
            traj.append(episode_data)
            traj_len.append(episode_step)
            # reset trajectory buffer
            data_, episode_step = defaultdict(list), 0

        episode_step += 1
    info = {
        "obs_mean": observations.mean(0, keepdims=True),
        "obs_std": observations.std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }

    return Dataset.create(
        traj=traj,
        info=info,
        # observations=observations,
        # actions=dataset['actions'].astype(np.float32),
        # rewards=dataset['rewards'].astype(np.float32),
        # masks=1.0 - dones_float.astype(np.float32),
        # dones_float=dones_float.astype(np.float32),
        # next_observations=next_observations,
    )



def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

def get_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    if 'antmaze' in env_name:
         return  dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset