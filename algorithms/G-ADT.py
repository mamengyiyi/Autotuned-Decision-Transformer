# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict, dataclass
import os
import random
import uuid

import d4rl  # noqa
import gym  # noqa
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa
import wandb
from pathlib import Path
from hiql import ImplicitQLearning, ValueFunction, DeterministicPolicy, GaussianPolicy
from wrapper import d4rl_utils

EXP_ADV_MAX = 100.0

class HIQLConfig:
    # Experiment
    device: str = "cuda:4"
    load_model: str = "/your/path/to/hiqlmodel"  # Model load file name, "" doesn't load
    # IQL
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 10.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    hiql_tau: float = 0.7  # Coefficient for asymmetric loss
    hiql_deterministic: bool = False  # Use deterministic actor
    max_timesteps: int = int(1e6)  # Max time steps to run environment

def load_hiql_model(state_dim, action_dim, config, max_action=1):


    v_network = ValueFunction(state_dim, state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, state_dim, action_dim, max_action)
        if config.hiql_deterministic
        else GaussianPolicy(state_dim, state_dim, action_dim, max_action)
    ).to(config.device)
    high_actor = GaussianPolicy(state_dim, state_dim, state_dim, max_action).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    high_actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

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
        # HIQL
        "beta": config.beta,
        "iql_tau": config.hiql_tau,
        "max_steps": config.max_timesteps,
        "use_waypoints": config.use_waypoints,
    }
    trainer = ImplicitQLearning(**kwargs)


    policy_file = Path(config.load_model)
    trainer.load_state_dict(torch.load(policy_file, map_location='cpu'))
    iql_value_func = trainer.vf.to(config.device)
    iql_high_actor = trainer.high_actor.to(config.device)
    iql_actor = trainer.actor.to(config.device)
    return iql_value_func, iql_high_actor, iql_actor


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "DT-D4RL"
    name: str = "DT"
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "antmaze-large-play-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_0000
    warmup_steps: int = 10_000
    reward_scale: float = 1.0
    num_workers: int = 1
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_episodes: int = 100
    eval_every: int = 50_000
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda:1"
    load_hiql_model: str = "/your/path/to/hiqlmodel"
    hiql_deterministic: bool = False
    load_bc_model: str = ""
    hiql_beta: float = 1.0
    hiql_tau: float = 0.7
    hiql_discount: float = 0.99
    use_waypoints: bool = True
    way_steps: int = 25

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
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


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    env = d4rl_utils.make_env(env_name)
    dataset = d4rl.qlearning_dataset(env)
    base_observation = dataset['observations'][0]



    traj, traj_len = [], []

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



    data_, episode_step = defaultdict(list), 0
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["next_observations"].append(dataset["next_observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["dones_float"].append(dones_float[i])


        if dones_float[i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_step)
            # reset trajectory buffer
            data_, episode_step = defaultdict(list), 0

        episode_step += 1



    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
        "base_observation": base_observation
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, way_steps=25):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.seq_len = seq_len
        self.way_steps = way_steps

        self.state_mean = 0.0
        # self.state_mean = info["obs_mean"]
        self.state_std = 1.0
        # self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        self.base_observation = info["base_observation"]
        # print(self.sample_prob)



    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]

        # terminal_locs, = np.nonzero(traj["dones_float"] > 0)
        # exit()
        # final_state_indx = terminal_locs[np.searchsorted(terminal_locs, indx)]
        traj['low_goals'] = np.ones_like(traj['observations']) * traj['observations'][-1]
        traj['low_goals'][0:-self.way_steps] = traj['observations'][self.way_steps:]


        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        next_states = traj["next_observations"][start_idx : start_idx + self.seq_len]
        low_goals = traj['low_goals'][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)



        states = (states - self.state_mean) / self.state_std
        next_states = (next_states - self.state_mean) / self.state_std
        low_goals = (low_goals - self.state_mean) / self.state_std
        # returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            next_states = pad_along_axis(next_states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            low_goals = pad_along_axis(low_goals, pad_to=self.seq_len)

        return states, actions, returns, low_goals, next_states, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_goal_emb = nn.Linear(state_dim + state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = state_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        goals: torch.Tensor,  # [batch_size, seq_len, state_dim]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        goal_state = torch.cat([goals, states], dim=-1)
        state_goal_emb = self.state_goal_emb(goal_state) + time_emb
        act_emb = self.action_emb(actions) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (g_0, s_0, a_0, g_1, s_1, a_1, ...)
        sequence = (
            torch.stack([state_goal_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_len, self.embedding_dim)
        )

        if padding_mask is not None:
            # padding_mask = padding_mask.reshape(batch_size, seq_len)
            padding_mask = (
                torch.stack([padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 2 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)

        # [batch_size, seq_len, action_dim]
        # predict actions only from goal-state embeddings
        out = self.action_head(out[:,0::2]) * self.max_action
        return out


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    # target_return: float,
    high_actor_func,
    base_observation,
    device: str = "cpu",
    use_waypoints: bool = True,
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    goals = torch.zeros(1, model.episode_len + 1,model.state_dim, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    init_state = torch.as_tensor(env.reset(), device=device)
    goal = env.wrapped_env.target_goal
    # print(goal)
    obs_goal = base_observation.copy()
    obs_goal[:2] = goal
    obs_goal = torch.as_tensor(obs_goal, device=device)
    states[:, 0] = init_state
    if not use_waypoints:
        cur_obs_goal = obs_goal
    else:
        cur_obs_goal = high_actor_func(init_state.float(), obs_goal).mean
        cur_obs_goal = init_state + cur_obs_goal

    goals[:, 0] = cur_obs_goal

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],  # noqa
            actions[:, : step + 1][:, -model.seq_len :],  # noqa
            goals[:, : step + 1][:, -model.seq_len :],  # noqa
            time_steps[:, : step + 1][:, -model.seq_len :],  # noqa
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        if not use_waypoints:
            next_obs_goal = obs_goal
            goals[:, step + 1] = next_obs_goal
        else:
            next_state = torch.as_tensor(next_state, device=device).float()
            next_obs_goal = high_actor_func(next_state, obs_goal).mean
            next_obs_goal = next_state + next_obs_goal
        # returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            goals[:, step + 1] = next_obs_goal.detach()

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    config.group = config.group.replace("iql_dt", "hiql_dt-onetoken-gs-a")
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name, seq_len=config.seq_len, way_steps=config.way_steps
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=d4rl_utils.make_env(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    HIQLConfig.device = config.device
    HIQLConfig.load_model = config.load_hiql_model
    HIQLConfig.hiql_deterministic = config.hiql_deterministic
    HIQLConfig.beta = config.hiql_beta
    HIQLConfig.use_waypoints = config.use_waypoints

    hiql_value_func, high_actor_func, actor_func = load_hiql_model(config.state_dim, config.action_dim, HIQLConfig)


    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, low_goals, next_states, time_steps, mask = [b.to(config.device) for b in batch]

        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)


        # loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        if config.use_waypoints:  # Use waypoint states as goals (for hierarchical policies)
            cur_goals = low_goals
        else:  # Use randomized last observations as goals (for flat policies)
            cur_goals = None

        predicted_actions = model(
            states=states,
            actions=actions,
            goals=cur_goals,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        v1, v2 = hiql_value_func.both(states, cur_goals)
        nv1, nv2 = hiql_value_func.both(next_states, cur_goals)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2

        adv = nv - v


        exp_adv = torch.exp(HIQLConfig.beta * adv.detach()).clamp(max=EXP_ADV_MAX).unsqueeze(-1)
        bc_losses = F.mse_loss(predicted_actions, actions, reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (exp_adv * bc_losses * mask.unsqueeze(-1)).mean()


        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )


        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            # for target_return in config.target_returns:
            eval_env.seed(config.eval_seed)
            eval_returns = []
            base_observation = dataset.base_observation
            for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                eval_return, eval_len = eval_rollout(
                    model=model,
                    env=eval_env,
                    high_actor_func=high_actor_func,
                    base_observation=base_observation,
                    device=config.device,
                    use_waypoints=config.use_waypoints,
                )
                # unscale for logging & correct normalized score computation
                eval_returns.append(eval_return / config.reward_scale)

            normalized_scores = (
                eval_env.get_normalized_score(np.array(eval_returns)) * 100
            )
            wandb.log(
                {
                    f"eval/return_mean": np.mean(eval_returns),
                    f"eval/return_std": np.std(eval_returns),
                    f"eval/normalized_score_mean": np.mean(
                        normalized_scores
                    ),
                    f"eval/normalized_score_std": np.std(
                        normalized_scores
                    ),
                },
                step=step,
            )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))



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
    episode_steps = []
    for i in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        step = 0
        while not done:
            step += 1
            if not use_waypoints:
                cur_obs_goal = obs_goal
            else:
                cur_obs_goal = high_actor.goal(state, obs_goal, device)

                cur_obs_goal = state + cur_obs_goal


            action = actor.act(state,cur_obs_goal, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        # print(i, step, episode_reward)
        episode_steps.append(step)
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.asarray(episode_steps)


if __name__ == "__main__":
    train()