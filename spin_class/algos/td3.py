from datetime import datetime
from re import A
import gym
import gym.spaces
import numpy as np
import os
import random
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any, Dict, List, Tuple, Union
import wandb

import spin_class.utils as utils


class SampleBatch:
    def __init__(
        self,
        state: np.array,
        action: np.array,
        reward: np.array,
        next_state: np.array,
        done: np.array,
        indices: np.array,
        device: torch.DeviceObjType = torch.device("cpu"),
        weights: np.array = None,
    ):
        self.state = torch.as_tensor(state, dtype=torch.float32, device=device)
        self.action = torch.as_tensor(action, dtype=torch.int64, device=device)
        self.reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        self.next_state = torch.as_tensor(
            next_state, dtype=torch.float32, device=device
        )
        self.done = torch.as_tensor(done, dtype=torch.float32, device=device)
        self.indices = indices
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        else:
            self.weights = None


class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size
        self.count = 0

    def add(
        self,
        state: np.array,
        action: np.array,
        reward: float,
        next_state: np.array,
        done: int,
    ) -> None:
        pass

    def sample(self, batch_size: int) -> SampleBatch:
        pass


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, env: gym.Env):
        super(UniformReplayBuffer, self).__init__(size)
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.state = np.zeros(size, dtype=np.int64)
            self.next_state = np.zeros(size, dtype=np.int64)
        elif isinstance(env.observation_space, gym.spaces.Box):
            shape = list(env.observation_space.sample().shape)
            self.state = np.zeros([size] + shape, dtype=np.float32)
            self.next_state = np.zeros([size] + shape, dtype=np.float32)
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action = np.zeros(size, dtype=np.int64)
        elif isinstance(env.action_space, gym.spaces.Box):
            shape = list(env.action_space.sample().shape)
            self.action = np.zeros([size] + shape, dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.tile(False, [size])

        self.ptr = 0

    def add(
        self,
        state: np.array,
        action: np.array,
        reward: float,
        next_state: np.array,
        done: int,
        *args,
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(
        self, batch_size: int, device: torch.DeviceObjType = torch.device("cpu")
    ) -> SampleBatch:
        indices = np.random.choice(self.count, size=batch_size)
        return SampleBatch(
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices],
            indices,
            device,
        )


def shape(space: gym.Space):
    if isinstance(space, gym.spaces.Discrete):
        return tuple()
    elif isinstance(space, gym.spaces.Box):
        return space.sample().shape
    else:
        raise Exception(f"Unsupported space type: {type(space)}")


def build_model(
    conf: List[Union[Tuple[str, int], Tuple[str, int, str]]],
    device: torch.DeviceObjType,
):
    layers = []
    t, input_shape, *_ = conf[0]
    assert t == "input"
    if isinstance(input_shape, tuple):
        prev_size = sum(input_shape)
        layers.append(Concat1d())
    else:
        prev_size = input_shape
    if len(conf) == 1:
        layers.append(nn.Identity())
    for layer in conf[1:]:
        if layer[0] == "linear":
            size = layer[1]
            layers.append(nn.Linear(prev_size, size))
            if layer[2] == "relu":
                layers.append(nn.ReLU())
            elif layer[2] == "none":
                pass
            elif layer[2] == "tanh":
                layers.append(nn.Tanh())
            else:
                raise NotImplementedError(f"Unrecognized activation type: {layer[2]}")
        elif layer[0] == "scaling":
            assert isinstance(layer[1], torch.Tensor)
            layers.append(ScaleLayer1d(layer[1]))
        elif layer[0] == "onehot":
            size = layer[1]
            layers.append(OneHot1d(size))
        elif layer[0] == "embed":
            num_classes = layer[1]
            size = layer[2]
            layers.append(
                nn.Embedding(
                    num_classes,
                    size,
                    sparse=False,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            raise ValueError(f"Unrecognized layer type: {layer[0]}")
        prev_size = size
    return nn.Sequential(*layers).to(device)


def gaussian_output_layers(env: gym.Env, device: torch.DeviceObjType):
    signal_count = env.action_space.shape[0]

    output_layers = [("linear", signal_count, "tanh")]

    # TOOD: Support infinite sized boxes.
    signal_scale = torch.as_tensor(
        (env.action_space.high - env.action_space.low) / 2.0,
        dtype=torch.float32,
        device=device,
    )
    if not torch.all(torch.isclose(signal_scale, torch.ones_like(signal_scale))):
        output_layers.append(("scaling", signal_scale))

    if not np.all(
        np.isclose(env.action_space.high, -env.action_space.low, equal_nan=True)
    ):
        # TODO: add offset layer.
        raise NotImplementedError(
            "Box ranges which are not centered at 0 are not yet implemented."
        )

    return output_layers


class ScaleLayer1d(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super(ScaleLayer1d, self).__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.scale)


class OneHot1d(nn.Module):
    def __init__(self, num_classes: int):
        super(OneHot1d, self).__init__()

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        y = torch.as_tensor(x, dtype=torch.int64)
        return F.one_hot(y, num_classes=self.num_classes).to(torch.float32)


class Concat1d(nn.Module):
    def __init__(self):
        super(Concat1d, self).__init__()

    def forward(self, args: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.cat(args, dim=1)


class TD3Model(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.DeviceObjType,
    ):
        super(TD3Model, self).__init__()

        self.env = env
        self.device = device
        self.layers = layers
        self.head = build_model(layers, device)

    def forward(self, x: torch.Tensor):
        return self.head(x)


class TD3ActorMLP(TD3Model):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        std_logits: float = -0.5,
        std_decay: bool = False,
    ):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        shape = env.observation_space.sample().shape[0]
        layers = [("input", shape)]
        layers += [("linear", layer_size, activation)] * num_layers
        layers += gaussian_output_layers(env, device)
        super(TD3ActorMLP, self).__init__(env, layers, device)

        signal_count = env.action_space.shape[0]
        self.cov = torch.diag(
            torch.exp(
                std_logits
                * torch.ones((signal_count,), dtype=torch.float32, device=device)
            )
        )
        self.std_decay = std_decay

    def forward(self, x: torch.Tensor):
        return self.head(x)

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        k = np.exp(-3 * c) if self.std_decay else 1
        return MultivariateNormal(output, self.cov * k)


class TD3CriticMLP(TD3Model):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
    ):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        obs_shape = env.observation_space.sample().shape[0]
        act_shape = env.action_space.sample().shape[0]
        layers = [("input", (obs_shape, act_shape))]
        layers += [("linear", layer_size, activation)] * num_layers
        layers += [("linear", 1, "none")]
        super(TD3CriticMLP, self).__init__(env, layers, device)

    def forward(self, x: torch.Tensor):
        return self.head(x)


def make_models(
    env: gym.Env, device: torch.DeviceObjType, config: Dict[str, Any]
) -> Tuple[TD3Model, TD3Model]:
    return (
        TD3ActorMLP(
            env,
            config["pi_num_layers"],
            config["pi_layer_size"],
            config["pi_activation"],
            device,
            config["pi_std_logits"],
            config["pi_std_decay"],
        ),
        TD3CriticMLP(
            env,
            config["q_num_layers"],
            config["q_layer_size"],
            config["q_activation"],
            device,
        ),
        TD3CriticMLP(
            env,
            config["q_num_layers"],
            config["q_layer_size"],
            config["q_activation"],
            device,
        ),
    )


def train(
    env: gym.Env,
    config: Dict[str, Any],
    device: torch.DeviceObjType,
    run_id: str,
    run_name: str,
):
    model_dir = f"models/td3/{env.spec.id.lower()}"
    os.makedirs(f"{model_dir}", mode=0o755, exist_ok=True)

    def save(pi_net, q1_net, q2_net, step):
        dt_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        state = {
            "config": config,
            "pi_state_dict": pi_net.state_dict(),
            "q1_state_dict": q1_net.state_dict(),
            "q2_state_dict": q2_net.state_dict(),
        }
        torch.save(
            state,
            f"{model_dir}/{run_name}_{run_id}_{step}_{dt_str}.pth",
        )

    config = utils.add_defaults("td3", config)

    # Make the training reproducible.
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])
    env.observation_space.seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

    pi_online, q1_online, q2_online = make_models(env, device, config)
    ema_avg = (
        lambda averaged_model_parameter, model_parameter, num_averaged: config["rho"]
        * averaged_model_parameter
        + (1 - config["rho"]) * model_parameter
    )
    pi_target = torch.optim.swa_utils.AveragedModel(
        pi_online, device=device, avg_fn=ema_avg
    )
    q1_target = torch.optim.swa_utils.AveragedModel(
        q1_online, device=device, avg_fn=ema_avg
    )
    q2_target = torch.optim.swa_utils.AveragedModel(
        q2_online, device=device, avg_fn=ema_avg
    )

    wandb.watch(pi_online, log="all", log_freq=1024)
    wandb.watch(q1_online, log="all", log_freq=1024)
    wandb.watch(q2_online, log="all", log_freq=1024)

    pi_online_opt = Adam(pi_online.parameters(), lr=config["lr"])
    q1_online_opt = Adam(q1_online.parameters(), lr=config["lr"])
    q2_online_opt = Adam(q2_online.parameters(), lr=config["lr"])

    save_max_eps = config["save_max_eps"]
    save_final = config["save_final"]
    gamma = torch.as_tensor(config["gamma"], dtype=torch.float32, device=device)
    batch_size = config["batch_size"]
    log_step = config["log_step"]
    avg_eps_len = 0
    max_avg_eps_rew = float("-inf")
    learning_starts = config["learning_starts"]
    training_freq = config["training_freq"]
    training_count = config["training_count"]
    policy_freq = config["policy_freq"]
    buffer = UniformReplayBuffer(config["buffer_size"], env)
    total_steps = config["steps"]
    c = config["max_noise"]

    obs_dtype = (
        torch.int64
        if isinstance(env.observation_space, gym.spaces.Discrete)
        else torch.float32
    )

    done = False
    max_performance = False
    eps_len = 0
    eps_lens = []
    total_rew = 0
    total_rews = []
    q1_losses = []
    q2_losses = []
    pi_losses = []
    obs = env.reset()
    for step in range(1, total_steps + 1):
        eps_len += 1

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=obs_dtype, device=device).unsqueeze(0)
            mu = pi_online(obs_t)
            dist = pi_online.distribution(mu, step / total_steps)
            a = dist.sample()[0].cpu().numpy().tolist()
            next_obs, r, done, _ = env.step(a)
            buffer.add(obs, a, r, next_obs, 1 if done else 0)
            total_rew += r
            obs = next_obs

        if done:
            done = False
            total_rews.append(total_rew)
            eps_lens.append(eps_len)
            eps_len = 0
            total_rew = 0
            obs = env.reset()

        if step % training_freq == 0 and step >= learning_starts:
            for i in range(training_count):
                batch = buffer.sample(batch_size, device)
                q1_online_opt.zero_grad()
                q2_online_opt.zero_grad()
                with torch.no_grad():
                    m = pi_target(batch.next_state)
                    dist = pi_target.module.distribution(mu, step / total_steps)
                    action = torch.maximum(torch.minimum(dist.sample(), m + c), m - c)
                    if len(action.shape) == 1:
                        action = action.unsqueeze(-1)
                    q = torch.minimum(
                        q1_target((batch.next_state, action)),
                        q2_target((batch.next_state, action)),
                    ).squeeze()
                    y = batch.reward + gamma * (1 - batch.done) * q
                delta1 = torch.abs(y - q1_online((batch.state, batch.action)).squeeze())
                delta2 = torch.abs(y - q2_online((batch.state, batch.action)).squeeze())
                q1_loss = (delta1 ** 2).mean()
                q2_loss = (delta2 ** 2).mean()
                q1_loss.backward()
                q2_loss.backward()
                q1_losses.append(q1_loss.item())
                q2_losses.append(q2_loss.item())
                q1_online_opt.step()
                q2_online_opt.step()

                if i % policy_freq == 0:
                    q1_online_opt.zero_grad()
                    pi_online_opt.zero_grad()
                    pi = pi_online(batch.state)
                    if len(pi.shape) == 1:
                        pi = pi.unsqueeze(-1)
                    pi_loss = -q1_online((batch.state, pi)).mean()
                    pi_loss.backward()
                    pi_losses.append(pi_loss.item())
                    pi_online_opt.step()

                    pi_target.update_parameters(pi_online)
                    q1_target.update_parameters(q1_online)
                    q2_target.update_parameters(q2_online)

        if step % log_step == 0 and step > 0:
            avg_eps_rew = sum(total_rews) / (len(total_rews) if total_rews else 1)
            max_eps_rew = max(total_rews) if total_rews else 0
            min_eps_rew = min(total_rews) if total_rews else 0
            std_eps_rew = np.std(total_rews).tolist() if total_rews else 0
            max_avg_eps_rew = max(max_avg_eps_rew, avg_eps_rew)
            avg_eps_len = sum(eps_lens) / (len(eps_lens) if eps_lens else 1)
            max_eps_len = max(eps_lens) if eps_lens else 0
            min_eps_len = min(eps_lens) if eps_lens else 0
            std_eps_len = np.std(eps_lens).tolist() if eps_lens else 0
            avg_q1_loss = sum(q1_losses) / (len(q1_losses) if q1_losses else 1)
            avg_q2_loss = sum(q2_losses) / (len(q2_losses) if q2_losses else 1)
            avg_pi_loss = sum(pi_losses) / (len(pi_losses) if pi_losses else 1)
            total_rews = []
            eps_lens = []
            q_losses = []
            wandb.log(
                {
                    "avg_eps_rew": avg_eps_rew,
                    "max_eps_rew": max_eps_rew,
                    "min_eps_rew": min_eps_rew,
                    "std_eps_rew": std_eps_rew,
                    "max_avg_eps_rew": max_avg_eps_rew,
                    "avg_eps_len": avg_eps_len,
                    "max_eps_len": max_eps_len,
                    "min_eps_len": min_eps_len,
                    "std_eps_len": std_eps_len,
                    "pi_loss": avg_pi_loss,
                    "q1_loss": avg_q1_loss,
                    "q2_loss": avg_q2_loss,
                    "steps": step,
                }
            )

            print(
                ", ".join(
                    [
                        f"steps: {step}",
                        f"avg total rew: {avg_eps_rew:.4f}",
                        f"max total rew: {max_eps_rew:.2f}",
                        f"min total rew: {min_eps_rew:.2f}",
                        f"avg eps length: {avg_eps_len:.2f}",
                        f"pi loss: {avg_pi_loss:.6f}",
                        f"q1 loss: {avg_q1_loss:.6f}",
                        f"q2 loss: {avg_q2_loss:.6f}",
                    ]
                )
            )

            if save_max_eps:
                if int(min_eps_len) == int(env.spec.max_episode_steps):
                    if not max_performance:
                        save(pi_online, q1_online, q2_online, step)
                    max_performance = True
                else:
                    max_performance = False

    if save_final:
        save(pi_online, q1_online, q2_online, step)
