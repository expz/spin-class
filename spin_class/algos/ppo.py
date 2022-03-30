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


def shape(space: gym.Space):
    if isinstance(space, gym.spaces.Discrete):
        return tuple()
    elif isinstance(space, gym.spaces.Box):
        return space.sample().shape
    else:
        raise Exception(f"Unsupported space type: {type(space)}")


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


class PPOModel(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.DeviceObjType,
    ):
        super(PPOModel, self).__init__()

        self.env = env
        self.device = device
        self.layers = layers
        self.head = build_model(layers, device)

    def forward(self, x: torch.Tensor):
        return self.head(x)


class PPOTrunkMLP(PPOModel):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        embed_size: int = 0,
    ):
        if isinstance(env.observation_space, gym.spaces.Box):
            layers = [("input", shape(env.observation_space)[0])]
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            if embed_size > 0:
                layers = [("embed", env.observation_space.n, embed_size)]
            else:
                layers = [("onehot", env.observation_space.n)]
            layers = [("input", 1)] + layers
        layers += [("linear", layer_size, activation)] * num_layers

        super(PPOTrunkMLP, self).__init__(env, layers, device)

        self.output_size = layer_size


class PPOValueMLP(PPOModel):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        embed_size: int = 0,
    ):
        layers = [("input", layer_size), ("linear", 1, "none")]
        super(PPOValueMLP, self).__init__(env, layers, device)

        self.trunk = PPOTrunkMLP(
            env, num_layers, layer_size, activation, device, embed_size
        )

    def forward(self, x: torch.tensor):
        return self.head(self.trunk(x))


class PPOCategoricalPolicyMLP(PPOModel):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        embed_size: int = 0,
    ):
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.num_layers = num_layers
        self.layer_size = layer_size
        self.activation = activation
        self.embed_size = embed_size

        layers = [("input", layer_size), ("linear", env.action_space.n, "none")]
        super(PPOCategoricalPolicyMLP, self).__init__(env, layers, device)

        self.trunk = PPOTrunkMLP(
            env, num_layers, layer_size, activation, device, embed_size
        )

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        return Categorical(logits=output)

    def forward(self, x: torch.Tensor):
        return self.head(self.trunk(x))


def make_models(
    env: gym.Env, device: torch.DeviceObjType, config: Dict[str, Any]
) -> Tuple[PPOModel, PPOModel]:
    return (
        PPOCategoricalPolicyMLP(
            env,
            config["pi_num_layers"],
            config["pi_layer_size"],
            config["pi_activation"],
            device,
            config["pi_embed_size"],
        ),
        PPOValueMLP(
            env,
            config["vf_num_layers"],
            config["vf_layer_size"],
            config["vf_activation"],
            device,
        ),
    )


def clone(model: PPOModel):
    if isinstance(model, PPOCategoricalPolicyMLP):
        clone = PPOCategoricalPolicyMLP(
            model.env,
            model.num_layers,
            model.layer_size,
            model.activation,
            model.device,
            model.embed_size,
        )
        clone.load_state_dict(model.state_dict())
    else:
        raise Exception(f"Unsupported model type for cloning: {type(model)}")
    return clone


def train(
    env: gym.Env,
    config: Dict[str, Any],
    device: torch.DeviceObjType,
    run_id: str,
    run_name: str,
):
    model_dir = f"models/ppo/{env.spec.id.lower()}"
    os.makedirs(f"{model_dir}", mode=0o755, exist_ok=True)

    def save(pi_net, q_net, step):
        dt_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        state = {
            "config": config,
            "pi_state_dict": pi_net.state_dict(),
            "q_state_dict": q_net.state_dict(),
        }
        torch.save(
            state,
            f"{model_dir}/{run_name}_{run_id}_{step}_{dt_str}.pth",
        )

    config = utils.add_defaults("ppo", config)

    # Make the training reproducible.
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])
    env.observation_space.seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

    pi, vf = make_models(env, device, config)

    wandb.watch(pi, log="all", log_freq=1024)
    wandb.watch(vf, log="all", log_freq=1024)

    pi_opt = Adam(pi.parameters(), lr=config["lr"])
    vf_opt = Adam(vf.parameters(), lr=config["lr"])

    save_final = config["save_final"]
    gamma = torch.as_tensor(config["gamma"], dtype=torch.float32, device=device)
    lam = config["lambda"]
    batch_size = config["batch_size"]
    log_step = config["log_step"]
    avg_eps_len = 0
    max_avg_eps_rew = float("-inf")
    total_steps = config["total_steps"]
    epsilon = config["epsilon"]
    iter_steps = config["iter_steps"]
    epochs = config["epochs"]
    c1 = config["vf_loss_coeff"]
    c2 = config["entropy_loss_coeff"]

    assert iter_steps % batch_size == 0, "iter_steps must be divisible by batch_size"

    obs_dtype = (
        torch.int64
        if isinstance(env.observation_space, gym.spaces.Discrete)
        else torch.float32
    )

    max_performance = False
    vf_losses = []
    pi_losses = []
    obs = env.reset()
    step = 0
    for _ in range(total_steps // iter_steps):
        done = False
        obs = env.reset()
        obss = np.zeros([iter_steps] + list(shape(env.observation_space)))
        if len(obss.shape) == 1:
            obss = np.expand_dims(obss, axis=1)
        rets = torch.zeros(iter_steps, dtype=torch.float32, device=device)
        advs = torch.zeros(iter_steps, dtype=torch.float32, device=device)
        as_: torch.Tensor = torch.zeros(
            [iter_steps] + list(shape(env.action_space)),
            dtype=torch.int64,
            device=device,
        )
        if len(as_.shape) == 1:
            as_ = as_.unsqueeze(1)
        rs = torch.zeros(iter_steps, dtype=torch.float32, device=device)
        eps_vs = torch.zeros(iter_steps, dtype=torch.float32, device=device)
        eps_rs = torch.zeros(iter_steps, dtype=torch.float32, device=device)
        ptr = 0
        eps_len = 0
        eps_lens = []
        total_rew = 0
        total_rews = []
        first_step = True
        while first_step or step % iter_steps != 0:
            first_step = False

            i = step % iter_steps

            eps_len += 1

            obss[i, :] = obs

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=obs_dtype, device=device)
                obs_t = obs_t.unsqueeze(0)
                p = pi(obs_t)[0]
                v = vf(obs_t)[0]
                dist = pi.distribution(p, step / total_steps)
                a = dist.sample()
                next_obs, r, done, _ = env.step(a.cpu().numpy().tolist())

            eps_vs[eps_len - 1] = v
            as_[i, :] = a
            eps_rs[eps_len - 1] = r
            rs[i] = r

            total_rew += r
            obs = next_obs

            if done or (step + 1) % iter_steps == 0:
                if done or not total_rews:
                    eps_lens.append(eps_len)
                    total_rews.append(total_rew)
                ret = 0
                for j in range(eps_len - 1, -1, -1):
                    ret = eps_rs[j] + gamma * ret
                    rets[ptr + j] = ret
                adv = eps_rs[-1] - eps_vs[-1]
                advs[ptr + eps_len - 1] = adv
                for j in range(eps_len - 2, -1, -1):
                    adv = (
                        eps_rs[j] + gamma * eps_vs[j + 1] - eps_vs[j]
                    ) + lam * gamma * adv
                    advs[ptr + j] = adv
                ptr += eps_len
                done = False
                eps_len = 0
                total_rew = 0
                obs = env.reset()

            step += 1

        avg_eps_rew = sum(total_rews) / len(total_rews)
        max_eps_rew = max(total_rews)
        min_eps_rew = min(total_rews)
        std_eps_rew = np.std(total_rews).tolist()
        max_avg_eps_rew = max(max_avg_eps_rew, avg_eps_rew)
        avg_eps_len = sum(eps_lens) / len(eps_lens)
        max_eps_len = max(eps_lens)
        min_eps_len = min(eps_lens)
        std_eps_len = np.std(eps_lens).tolist()

        obs = torch.as_tensor(obss.squeeze(), dtype=obs_dtype, device=device)
        a = as_.squeeze()
        ret = rets
        std, mean = advs.std(dim=0), advs.mean()
        adv = (advs - mean) / std

        pi_k = clone(pi)

        for k in range(epochs):
            for j in range(iter_steps // batch_size):
                obs_b = obs[j * batch_size : (j + 1) * batch_size]
                a_b = a[j * batch_size : (j + 1) * batch_size]
                ret_b = ret[j * batch_size : (j + 1) * batch_size]
                adv_b = adv[j * batch_size : (j + 1) * batch_size]

                pi_opt.zero_grad()
                vf_opt.zero_grad()

                # Calculate pi loss
                dist = pi.distribution(pi(obs_b), step / total_steps)
                p_b = dist.probs.gather(-1, a_b.unsqueeze(-1))
                p_b = p_b.squeeze()
                assert len(p_b.shape) == 1
                dist_k = pi.distribution(pi_k(obs_b), step / total_steps)
                p_k_b = dist_k.probs.gather(-1, a_b.unsqueeze(-1))
                p_k_b = p_k_b.squeeze()
                assert len(p_k_b.shape) == 1
                g = torch.where(
                    adv_b >= 0, (1 + epsilon) * adv_b, (1 - epsilon) * adv_b
                )
                pi_loss = -torch.minimum((p_b / p_k_b) * adv_b, g).mean()

                # Calculate vf loss
                v_b = vf(obs_b).squeeze()
                vf_loss = ((v_b - ret_b) ** 2).mean()

                # Calculate entropy loss
                avg_entropy = dist.entropy().sum() / obs_b.shape[0]
                entropy_loss = -avg_entropy

                # Calculate total loss
                loss = pi_loss + c1 * vf_loss + c2 * entropy_loss
                loss.backward()

                pi_opt.step()
                vf_opt.step()

        if step % log_step == 0:
            wandb.log(
                {
                    "avg_entropy": avg_entropy.item(),
                    "avg_eps_rew": avg_eps_rew,
                    "max_eps_rew": max_eps_rew,
                    "min_eps_rew": min_eps_rew,
                    "std_eps_rew": std_eps_rew,
                    "max_avg_eps_rew": max_avg_eps_rew,
                    "avg_eps_len": avg_eps_len,
                    "max_eps_len": max_eps_len,
                    "min_eps_len": min_eps_len,
                    "std_eps_len": std_eps_len,
                    "pi_loss": pi_loss.item(),
                    "vf_loss": vf_loss.item(),
                    "total_loss": loss.item(),
                    "adv_std": std,
                    "adv_mean": mean,
                    "steps": step,
                }
            )

        print(
            ", ".join(
                [
                    f"steps: {step}",
                    f"avg total rew: {avg_eps_rew:.4f}",
                    f"entropy: {avg_entropy.item():.4f}",
                    f"max eps length: {max_eps_len:.2f}",
                    f"min eps length: {min_eps_len}",
                    f"pi loss: {pi_loss.item():.6f}",
                    f"vf_loss: {vf_loss:.6f}",
                    f"loss: {loss.item():.6f}",
                ]
            )
        )

        if np.isclose(avg_entropy.item(), 0.0):
            print("Stopping early: entropy has gone to zero.")
            break

    if save_final:
        save(pi, vf, step)
