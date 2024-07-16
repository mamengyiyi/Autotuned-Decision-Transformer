# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from wrapper import d4rl_utils

TensorBatch = List[torch.Tensor]
os.environ["http_proxy"] = "http://172.19.135.130:5000"
os.environ["https_proxy"] = "http://172.19.135.130:5000"
os.environ["WANDB_API_KEY"] = '3d1ee35f09ab325a640e4543c0924b73e974de24'
os.environ["WANDB_MODE"] = 'online'

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

from pathlib import Path
@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda:1"
    env: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = 'None'   # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 1024  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 1.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = True  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    use_waypoints: bool = True
    way_steps: int = 25
    # Wandb logging
    project: str = "HIQL"
    group: str = "HIQL-D4RL"
    name: str = "HIQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{self.way_steps}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    # target = env.wrapped_env.target_goal
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        env_name: str = "antmaze-large-play-v2",
        device: str = "cpu",

    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.p_randomgoal = 0.1 # 0.3
        self.p_trajgoal = 0.7  # 0.5
        self.p_currgoal = 0.2  # 0.2
        self.high_p_randomgoal = 0.1  # 0.3
        self.env_name = env_name

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._goals = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        # self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])

        if 'antmaze' in self.env_name:
            # antmaze: terminals are incorrect for GCRL
            dones_float = np.zeros_like(data['rewards'])
            data['terminals'][:] = 0.

            for i in range(len(dones_float) - 1):
                if np.linalg.norm(data['observations'][i + 1] - data['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
        else:
            dones_float = data['terminals'].copy()

        self._dones[:n_transitions] = self._to_tensor(dones_float[..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        self.terminal_locs,  = np.nonzero(dones_float > 0)


        print(f"Dataset size: {n_transitions}")


    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None, geom_sample=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal


        batch_size = len(indx)
        # print(batch_size)
        # Random goals
        goal_indx = np.random.randint(min(self._size, self._pointer), size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(0.99)).astype(int),
                                          final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, way_steps=10) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        goal_indx = self.sample_goals(indices, geom_sample=1)

        states = self._states[indices]
        actions = self._actions[indices]
        # rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        goals = self._states[goal_indx]

        success = (indices == goal_indx)
        rewards = torch.from_numpy(success.astype(float)).to(states.device)

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indices)]
        way_indx = np.minimum(indices + way_steps, final_state_indx)
        low_goals = self._states[way_indx]
        distance = np.random.rand(batch_size)

        high_traj_goal_indx = np.round(
            (np.minimum(indices + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indices + way_steps, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self._size, size=batch_size)
        high_random_target_indx = np.minimum(indices + way_steps, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        high_goals = self._states[high_goal_idx]
        high_targets = self._states[high_target_idx]



        return [states, actions, rewards, next_states, dones, goals, low_goals, high_goals, high_targets]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, high_actor: nn.Module, base_observation, device: str, n_episodes: int, seed: int, use_waypoints=False,
) -> np.ndarray:
    env.seed(seed)

    goal = env.wrapped_env.target_goal
    obs_goal = base_observation.copy()
    obs_goal[:2] = goal
    actor.eval()
    episode_rewards = []
    for i in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            if not use_waypoints:
                cur_obs_goal = obs_goal
            else:
                cur_obs_goal = high_actor.goal(state, obs_goal, device)

                cur_obs_goal = state + cur_obs_goal


            action = actor.act(state,cur_obs_goal, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


# def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
#     return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def expectile_loss(adv, diff, expectile): # equals to asymmetric_l2_loss
    weight = torch.where(adv > 0, expectile, (1 - expectile))
    return (weight * (diff**2)).mean()


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        dropout_rate: float = None,
        layer_norm: bool = False,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation_fn())
            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[-1]))
            layers.append(output_activation_fn())
            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 512,
        n_hidden: int = 2,
        output_activation_tanh: bool = True
    ):
        super().__init__()
        self.net = MLP(
            [state_dim+goal_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh if output_activation_tanh else None
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> MultivariateNormal:
        sg = torch.cat([obs, goal], dim=-1)

        mean = self.net(sg)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: np.ndarray, goal: np.array, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        goal = torch.tensor(goal.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state, goal)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

    @torch.no_grad()
    def goal(self, state: np.ndarray, goal: np.array, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        goal = torch.tensor(goal.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state, goal)
        action = dist.mean if not self.training else dist.sample()

        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 512,
        n_hidden: int = 2,
        output_activation_tanh: bool = True
    ):
        super().__init__()
        self.net = MLP(
            [state_dim+goal_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh if output_activation_tanh else None,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        sg = torch.cat([obs, goal], dim=-1)
        return self.net(sg)

    @torch.no_grad()
    def act(self, state: np.ndarray, goal: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        goal = torch.tensor(goal.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state, goal) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )

    @torch.no_grad()
    def goal(self, state: np.ndarray, goal: np.ndarray, device: str = "cpu"):

        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        goal = torch.tensor(goal.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state, goal).cpu().data.numpy().flatten()



class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 512, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, goal_dim: int, hidden_dim: int = 1024, n_hidden: int = 3):
        super().__init__()
        dims = [state_dim+goal_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = MLP(dims, layer_norm=True, squeeze_output=True)
        self.v2 = MLP(dims, layer_norm=True, squeeze_output=True)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:

        return torch.min(*self.both(state, goal))

    def both(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sg = torch.cat([state, goal], -1)
        return self.v1(sg), self.v2(sg)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        high_actor: nn.Module,
        high_actor_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        use_waypoints: bool = True,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.vf = v_network
        self.vf_target = copy.deepcopy(self.vf).requires_grad_(False).to(device)
        self.actor = actor
        self.high_actor = high_actor
        self.v_optimizer = v_optimizer
        self.actor_optimizer = actor_optimizer
        self.high_actor_optimizer = high_actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.high_actor_lr_schedule = CosineAnnealingLR(self.high_actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.use_waypoints = use_waypoints

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, rewards,  next_observations, terminals, goals, log_dict) -> torch.Tensor:
        # Update value function
        terminals = 1.0 - rewards
        rewards = rewards - 1.0
        with torch.no_grad():
            next_v1, next_v2 = self.vf_target.both(next_observations, goals)
            next_v = self.vf_target(next_observations, goals)

            v1_t,v2_t = self.vf_target.both(observations, goals)
            v_t = (v1_t + v2_t) / 2
        # print(terminals.shape)
        # print(rewards.shape)
        # print(next_v.shape)
        q = rewards + terminals * self.discount * next_v.detach()
        adv = q - v_t

        v1, v2 = self.vf.both(observations, goals)
        q1 = rewards + terminals * self.discount * next_v1.detach()
        q2 = rewards + terminals * self.discount * next_v2.detach()

        v_loss1 = expectile_loss(adv, q1 - v1, self.iql_tau)
        v_loss2 = expectile_loss(adv, q2 - v2, self.iql_tau)
        v_loss = v_loss1 + v_loss2

        log_dict["value_loss1"] = v_loss1.item()
        log_dict["value_loss2"] = v_loss2.item()
        log_dict["value_loss"] = v_loss.item()


        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        soft_update(self.vf_target, self.vf, self.tau)

    def _update_high_policy(self,batch, log_dict):
        states, actions, rewards, next_states, dones, goals, low_goals, high_goals, high_targets = batch
        cur_goals = high_goals
        v1, v2 = self.vf.both(states, cur_goals)
        nv1, nv2 = self.vf.both(high_targets, cur_goals)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.high_actor(states, high_goals)

        target = high_targets - states
        if isinstance(policy_out, torch.distributions.Distribution):

            bc_losses = -policy_out.log_prob(target)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != high_goals.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - target) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["high_actor_loss"] = policy_loss.item()
        self.high_actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.high_actor_optimizer.step()
        self.high_actor_lr_schedule.step()

    def _update_policy(self, batch, log_dict):

        states, actions, rewards, next_states, dones, goals, low_goals, high_goals, high_targets = batch
        if self.use_waypoints:  # Use waypoint states as goals (for hierarchical policies)
            cur_goals = low_goals
        else:  # Use randomized last observations as goals (for flat policies)
            cur_goals = high_goals
        v1, v2 = self.vf.both(states, cur_goals)
        nv1, nv2 = self.vf.both(next_states, cur_goals)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2

        adv = nv - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(states, cur_goals)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            goals, low_goals, high_goals, high_targets
        ) = batch
        log_dict = {}


        # Update value function
        self._update_v(observations, actions, rewards, next_observations, dones, goals, log_dict)

        # Update actor
        self._update_policy(batch, log_dict)
        self._update_high_policy(batch, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "high_actor": self.high_actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "high_actor_optimizer": self.high_actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # print(state_dict.keys())



        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        self.vf_target = copy.deepcopy(self.vf)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.high_actor.load_state_dict(state_dict["high_actor"])
        self.high_actor_optimizer.load_state_dict(state_dict["high_actor_optimizer"])

        self.total_it = state_dict["total_it"]



@pyrallis.wrap()
def train(config: TrainConfig):

    # env = gym.make(config.env)
    env = d4rl_utils.make_env(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)


    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.env,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    v_network = ValueFunction(state_dim, state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, state_dim, action_dim, max_action)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, state_dim, action_dim, max_action)
    ).to(config.device)
    high_actor = GaussianPolicy(state_dim, state_dim, state_dim, max_action, output_activation_tanh=False).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    high_actor_optimizer = torch.optim.Adam(high_actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "high_actor": high_actor,
        "high_actor_optimizer": high_actor_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
        "use_waypoints": config.use_waypoints,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size, way_steps=config.way_steps)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode

        if (t+1) % config.eval_freq == 0:
            base_observation = dataset['observations'][0]
            # print(base_observation)
            print(f"Time steps: {t}")
            eval_scores = eval_actor(
                env,
                actor,
                high_actor,
                base_observation,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                use_waypoints=config.use_waypoints
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None and (t+1) % (config.eval_freq) == 0:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            )


if __name__ == "__main__":
    train()
