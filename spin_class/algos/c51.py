from datetime import datetime
import gym
import gym.spaces
import numpy as np
import os
import random
import torch
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


def build_model(
    conf: List[Union[Tuple[str, int], Tuple[str, int, str]]],
    device: torch.DeviceObjType,
):
    layers = []
    t, prev_size, *_ = conf[0]
    assert t == "input"
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
            elif layer[2] == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError(f"Unrecognized activation type: {layer[2]}")
        elif layer[0] == "linear2d":
            x, y = layer[1]
            layers.append(Linear2d(prev_size, x, y))
            if layer[2] == "relu":
                layers.append(nn.ReLU())
            elif layer[2] == "none":
                pass
            elif layer[2] == "tanh":
                layers.append(nn.Tanh())
            elif layer[2] == "sigmoid":
                layers.append(nn.Sigmoid())
            elif layer[2] == "softmax":
                layers.append(nn.Softmax(dim=-1))
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


class Linear2d(nn.Module):
    def __init__(self, prev_layer_size: int, x: int, y: int):
        super(Linear2d, self).__init__()

        self.linear = nn.Linear(prev_layer_size, x * y)
        self.x = x
        self.y = y

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.reshape(self.linear(t), (-1, self.x, self.y))


class ScaleLayer1d(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super(ScaleLayer1d, self).__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.scale)


class OneHot1d(nn.Module):
    def __init__(self, num_classes: int):
        super(OneHot1d, self).__init__()

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, dtype=torch.int64)
        return F.one_hot(y, num_classes=self.num_classes).to(torch.float32)


class C51Model(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Union[Tuple[str, int], Tuple[str, int, str]]],
        device: torch.DeviceObjType,
    ):
        super(C51Model, self).__init__()

        self.env = env
        self.device = device
        self.head = build_model(layers, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class C51MLP(C51Model):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        dist_res: int,
        device: torch.DeviceObjType,
        embed_size: int = 0,
    ):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "DQN only support discrete action spaces"

        if isinstance(env.observation_space, gym.spaces.Box):
            shape = env.observation_space.sample().shape[0]
            layers = [("input", shape)]
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            layers = [("input", 1)]
            if embed_size > 0:
                layers += [("embed", env.observation_space.n, embed_size)]
            else:
                layers += [("onehot", env.observation_space.n)]
        layers += [("linear", layer_size, activation)] * num_layers

        layers += [("linear2d", (env.action_space.n, dist_res), "softmax")]

        super(C51MLP, self).__init__(env, layers, device)


def make_model(
    env: gym.Env, device: torch.DeviceObjType, config: Dict[str, Any]
) -> C51Model:
    return C51MLP(
        env,
        config["num_layers"],
        config["layer_size"],
        config["activation"],
        config["distribution_resolution"],
        device,
        config["embed_size"] if config["embed"] else 0,
    )


def train(
    env: gym.Env,
    config: Dict[str, Any],
    device: torch.DeviceObjType,
    run_id: str,
    run_name: str,
):
    model_dir = f"models/c51/{env.spec.id.lower()}"
    os.makedirs(f"{model_dir}", mode=0o755, exist_ok=True)

    def save(q_net, step):
        dt_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        state = {
            "config": config,
            "q_state_dict": q_net.state_dict(),
        }
        torch.save(
            state,
            f"{model_dir}/{run_name}_{run_id}_{step}_{dt_str}.pth",
        )

    def q_of_argmax(q):
        expected_q = torch.matmul(q, z)
        a = torch.argmax(expected_q, dim=1)
        a_idx = a.view(-1, 1, 1).expand(-1, 1, dist_res)
        return q.gather(dim=1, index=a_idx).view(batch_size, dist_res)

    config = utils.add_defaults("c51", config)

    # Make the training reproducible.
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])
    env.observation_space.seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

    q_online = make_model(env, device, config)
    q_target = make_model(env, device, config)
    q_target.load_state_dict(q_online.state_dict())

    wandb.watch(q_online, log="all", log_freq=1024)

    q_online_opt = Adam(q_online.parameters(), lr=config["lr"])

    save_max_eps = config["save_max_eps"]
    save_final = config["save_final"]
    gamma = torch.as_tensor(config["gamma"], dtype=torch.float32, device=device)
    batch_size = config["batch_size"]
    log_step = config["log_step"]
    avg_eps_len = 0
    max_avg_eps_rew = float("-inf")
    eps_sched_len = config["eps_sched_len"]
    min_eps = config["eps_sched_final"]
    learning_starts = config["learning_starts"]
    training_freq = config["training_freq"]
    use_target = config["use_target"]
    target_update_freq = config["target_update_freq"]
    max_return = torch.as_tensor(
        config["max_return"], dtype=torch.float32, device=device
    )
    min_return = torch.as_tensor(
        config["min_return"], dtype=torch.float32, device=device
    )
    dist_res = config["distribution_resolution"]
    min_idx = torch.as_tensor(0, dtype=torch.float32, device=device)
    max_idx = torch.as_tensor(dist_res - 1, dtype=torch.float32, device=device)
    buffer = UniformReplayBuffer(config["buffer_size"], env)

    def eps(step: int):
        return max(min_eps, min_eps + (1.0 - min_eps) * (1 - step / eps_sched_len))

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
    q_losses = []
    delta_z = torch.as_tensor(
        (max_return - min_return) / (dist_res - 1), dtype=torch.float32, device=device
    )
    z = torch.arange(
        min_return,
        max_return + 1e-3,
        delta_z,
        dtype=torch.float32,
        device=device,
    )
    m0 = torch.zeros([batch_size, dist_res], dtype=torch.float32, device=device)
    obs = env.reset()
    for step in range(1, config["steps"] + 1):
        eps_len += 1

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=obs_dtype, device=device).unsqueeze(0)
            q = q_online(obs_t)[0]
            expected_q = torch.matmul(q, z)
            if random.random() < eps(step):
                a = env.action_space.sample()
            else:
                a = torch.argmax(expected_q).cpu().numpy().tolist()
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
            batch = buffer.sample(batch_size, device)
            with torch.no_grad():
                if use_target:
                    q_next = q_target(batch.next_state)
                    q_next2 = q_online(batch.next_state)
                    expected_q_next2 = torch.matmul(q_next2, z)
                    a = torch.argmax(expected_q_next2, dim=1)
                    a_idx = a.view(-1, 1, 1).expand(-1, 1, dist_res)
                    q_next_a = q_next.gather(dim=1, index=a_idx).view(
                        batch_size, dist_res
                    )
                else:
                    q_next = q_online(batch.next_state)
                    q_next_a = q_of_argmax(q_next)
                m = m0.clone()
                r = batch.reward.view(-1, 1).expand(batch_size, dist_res)
                done = batch.done.view(-1, 1).expand(batch_size, dist_res)
                z_new = z.view(1, -1).expand(batch_size, dist_res)
                z_new = r + gamma * z_new * (1 - done)
                b = (z_new - min_return) / delta_z
                b = torch.minimum(torch.maximum(b, min_idx), max_idx)
                u, l = b.ceil(), b.floor()
                x1 = q_next_a * (u - b)
                x2 = torch.where(u == l, q_next_a, q_next_a * (b - l))
                u, l = u.long(), l.long()
                for j in range(dist_res):
                    m += torch.scatter(
                        m0.clone(),
                        dim=1,
                        index=l[:, j].view(-1, 1),
                        src=x1[:, j].view(-1, 1),
                    )
                    m += torch.scatter(
                        m0.clone(),
                        dim=1,
                        index=u[:, j].view(-1, 1),
                        src=x2[:, j].view(-1, 1),
                    )
            q_online_opt.zero_grad()
            q = q_online(batch.state)
            a_idx = batch.action.view(-1, 1, 1).expand(-1, 1, dist_res)
            q_a = q.gather(dim=1, index=a_idx).view(batch_size, dist_res)
            q_loss = -(m * torch.log(q_a)).sum(dim=1).mean()
            q_loss.backward()
            q_losses.append(q_loss.item())
            q_online_opt.step()

        if use_target and step % target_update_freq == 0:
            q_target.load_state_dict(q_online.state_dict())

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
            avg_q_loss = sum(q_losses) / (len(q_losses) if q_losses else 1)
            total_rews = []
            eps_lens = []
            q_losses = []
            wandb.log(
                {
                    "epsilon": eps(step),
                    "avg_eps_rew": avg_eps_rew,
                    "max_eps_rew": max_eps_rew,
                    "min_eps_rew": min_eps_rew,
                    "std_eps_rew": std_eps_rew,
                    "max_avg_eps_rew": max_avg_eps_rew,
                    "avg_eps_len": avg_eps_len,
                    "max_eps_len": max_eps_len,
                    "min_eps_len": min_eps_len,
                    "std_eps_len": std_eps_len,
                    "q_loss": avg_q_loss,
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
                        f"q loss: {avg_q_loss:.6f}",
                    ]
                )
            )

            if save_max_eps:
                if int(min_eps_len) == int(env.spec.max_episode_steps):
                    if not max_performance:
                        save(q_online, step)
                    max_performance = True
                else:
                    max_performance = False

    if save_final:
        save(q_online, step)
