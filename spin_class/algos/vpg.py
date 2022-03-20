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


class VPGModel(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.DeviceObjType,
    ):
        super(VPGModel, self).__init__()

        self.env = env
        self.device = device
        self.layers = layers
        self.head = build_model(layers, device)

    def forward(self, x: torch.Tensor):
        return self.head(x)


class VPGTrunkMLP(VPGModel):
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

        super(VPGTrunkMLP, self).__init__(env, layers, device)

        self.output_size = layer_size


class VPGValueMLP(VPGModel):
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
        super(VPGValueMLP, self).__init__(env, layers, device)

        self.trunk = VPGTrunkMLP(
            env, num_layers, layer_size, activation, device, embed_size
        )

    def forward(self, x: torch.tensor):
        return self.head(self.trunk(x))


class VPGValueMLPHead(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        trunk: VPGTrunkMLP,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
    ):
        layers = [("input", trunk.output_size)]
        layers += [("linear", layer_size, activation)] * num_layers
        layers += [("linear", 1, "none")]
        super(VPGValueMLPHead, self).__init__(env, layers, device)

        self.trunk = trunk

    def forward(self, x: torch.tensor):
        return self.head(self.trunk(x))


class VPGGaussianPolicyMLP(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        std_logits: float = -0.5,
        embed_size: int = 0,
    ):
        assert isinstance(env.action_space, gym.spaces.Box)

        output_layers = [("input", layer_size)]
        output_layers += gaussian_output_layers(env, device)
        super(VPGGaussianPolicyMLP, self).__init__(env, output_layers, device)

        signal_count = env.action_space.shape[0]
        self.cov = torch.diag(
            torch.exp(
                std_logits
                * torch.ones((signal_count,), dtype=torch.float32, device=device)
            )
        )

        self.trunk = VPGTrunkMLP(
            env, num_layers, layer_size, activation, device, embed_size
        )

    def forward(self, x: torch.Tensor):
        return self.head(self.trunk(x))

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        return MultivariateNormal(output, self.cov * np.exp(-3 * c))


class VPGGaussianPolicyMLPHead(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        trunk: VPGTrunkMLP,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
        std_logits: float = -0.5,
    ):
        assert isinstance(env.action_space, gym.spaces.Box)

        layers = [("input", trunk.output_size)]
        layers += [("linear", layer_size, activation)] * num_layers
        layers += gaussian_output_layers(env, device)
        super(VPGGaussianPolicyMLPHead, self).__init__(env, layers, device)

        signal_count = env.action_space.shape[0]
        self.cov = torch.diag(
            torch.exp(
                std_logits
                * torch.ones((signal_count,), dtype=torch.float32, device=device)
            )
        )

        self.trunk = trunk

    def forward(self, x: torch.Tensor):
        return self.head(self.trunk(x))

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        return MultivariateNormal(output, self.cov * np.exp(-3 * c))


class VPGCategoricalPolicyMLP(VPGModel):
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

        layers = [("input", layer_size), ("linear", env.action_space.n, "none")]
        super(VPGCategoricalPolicyMLP, self).__init__(env, layers, device)

        self.trunk = VPGTrunkMLP(
            env, num_layers, layer_size, activation, device, embed_size
        )

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        return Categorical(logits=output)

    def forward(self, x: torch.Tensor):
        return self.head(self.trunk(x))


class VPGCategoricalPolicyMLPHead(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        trunk: VPGTrunkMLP,
        num_layers: int,
        layer_size: int,
        activation: str,
        device: torch.DeviceObjType,
    ):
        assert isinstance(env.action_space, gym.spaces.Discrete)

        layers = [("input", trunk.output_size)]
        layers += [("linear", layer_size, activation)] * num_layers
        layers += [("linear", env.action_space.n, "none")]
        super(VPGCategoricalPolicyMLPHead, self).__init__(env, layers, device)

        self.trunk = trunk

    def distribution(self, output: torch.Tensor, c: float = 0.0):
        return Categorical(logits=output)

    def forward(self, x: torch.Tensor):
        return self.head(self.trunk(x))


def make_models(
    env: gym.Env, device: torch.DeviceObjType, config: Dict[str, Any]
) -> Tuple[VPGModel, VPGModel]:
    if config["trunk_shared"]:
        trunk = VPGTrunkMLP(
            env,
            config["trunk_num_layers"],
            config["trunk_layer_size"],
            config["trunk_activation"],
            device,
            config["trunk_embed_size"] if config["trunk_embed"] else 0,
        )
        if isinstance(env.action_space, gym.spaces.Box):
            pi = VPGGaussianPolicyMLPHead(
                env,
                trunk,
                config["pi_num_layers"],
                config["pi_layer_size"],
                config["pi_activation"],
                device,
                config["std_logits"],
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            pi = VPGCategoricalPolicyMLPHead(
                env,
                trunk,
                config["pi_num_layers"],
                config["pi_layer_size"],
                config["pi_activation"],
                device,
            )
        else:
            raise NotImplementedError(
                f"Action space type not yet supported: {type(env.action_space)}"
            )
        vf = VPGValueMLPHead(
            env,
            trunk,
            config["vf_num_layers"],
            config["vf_layer_size"],
            config["vf_activation"],
            device,
        )
    else:
        if isinstance(env.action_space, gym.spaces.Box):
            pi = VPGGaussianPolicyMLP(
                env,
                config["pi_num_layers"],
                config["pi_layer_size"],
                config["pi_activation"],
                device,
                config["pi_embed_size"] if config["pi_embed"] else 0,
                config["std_logits"],
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            pi = VPGCategoricalPolicyMLP(
                env,
                config["pi_num_layers"],
                config["pi_layer_size"],
                config["pi_activation"],
                device,
                config["pi_embed_size"] if config["pi_embed"] else 0,
            )
        else:
            raise NotImplementedError(
                f"Action space type not yet supported: {type(env.action_space)}"
            )
        vf = VPGValueMLP(
            env,
            config["vf_num_layers"],
            config["vf_layer_size"],
            config["vf_activation"],
            device,
            config["vf_embed_size"] if config["vf_embed"] else 0,
        )
    return pi, vf


def train(
    env: gym.Env,
    config: Dict[str, Any],
    device: torch.DeviceObjType,
    run_id: str,
    run_name: str,
):
    model_dir = f"models/vpg/{env.spec.id.lower()}"
    os.makedirs(f"{model_dir}", mode=0o755, exist_ok=True)

    def save(pi, vf, step):
        dt_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        state = {
            "config": config,
            "pi_state_dict": pi.state_dict(),
            "vf_state_dict": vf.state_dict(),
        }
        torch.save(
            state,
            f"{model_dir}/{run_name}_{run_id}_{step}_{dt_str}.pth",
        )

    config = utils.add_defaults("vpg", config)

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

    wandb.watch(pi)
    wandb.watch(vf)

    pi_opt = Adam(pi.parameters(), lr=config["pi_lr"])
    vf_opt = Adam(vf.parameters(), lr=config["vf_lr"])

    save_max_eps = config["save_max_eps"]
    save_final = config["save_final"]
    gamma = config["gamma"]
    lam = config["lambda"]
    batch_size = config["batch_size"]
    avg_eps_len = 0
    max_avg_eps_rew = float("-inf")
    max_performance = False
    epsilon = 1e-6
    eta = config["entropy_eta"]

    obs_dtype = (
        torch.int64
        if isinstance(env.observation_space, gym.spaces.Discrete)
        else torch.float32
    )

    for k in range(0, config["steps"], batch_size):
        done = False
        obs = env.reset()
        obss = np.zeros([batch_size] + list(shape(env.observation_space)))
        if len(obss.shape) == 1:
            obss = np.expand_dims(obss, axis=1)
        rets = torch.zeros(batch_size, dtype=torch.float32, device=device)
        advs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        as_: torch.Tensor = torch.zeros(
            [batch_size] + list(shape(env.action_space)),
            dtype=torch.int64,
            device=device,
        )
        if len(as_.shape) == 1:
            as_ = as_.unsqueeze(1)
        rs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        eps_vs, eps_rs = torch.zeros(
            batch_size, dtype=torch.float32, device=device
        ), torch.zeros(batch_size, dtype=torch.float32, device=device)
        ptr = 0
        eps_lens = []
        eps_len = 0
        total_rews = []
        total_rew = 0
        for i in range(batch_size):
            eps_len += 1

            obss[i, :] = obs

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=obs_dtype, device=device).unsqueeze(
                    0
                )
                p = pi(obs_t)[0]
                v = vf(obs_t)[0]
                dist = pi.distribution(p, k / config["steps"])
                a = dist.sample()
                obs, r, done, _ = env.step(a.cpu().numpy().tolist())

            eps_vs[eps_len - 1] = v
            as_[i, :] = a
            eps_rs[eps_len - 1] = r
            rs[i] = r

            total_rew += r

            if done or i == batch_size - 1:
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

        step = k + batch_size

        avg_eps_rew = sum(total_rews) / len(total_rews)
        max_eps_rew = max(total_rews)
        min_eps_rew = min(total_rews)
        std_eps_rew = np.std(total_rews).tolist()
        max_avg_eps_rew = max(max_avg_eps_rew, avg_eps_rew)
        avg_eps_len = sum(eps_lens) / len(eps_lens)
        max_eps_len = max(eps_lens)
        min_eps_len = min(eps_lens)
        std_eps_len = np.std(eps_lens).tolist()

        # Save models whenever max performance is reached
        if save_max_eps:
            if int(min_eps_len) == int(env.spec.max_episode_steps):
                if not max_performance:
                    save(pi, vf, step)
                max_performance = True
            else:
                max_performance = False
        elif save_final and step >= config["steps"]:
            save(pi, vf, step)

        obs_b = torch.as_tensor(obss.squeeze(), dtype=obs_dtype, device=device)
        a_b = as_.squeeze()
        ret_b = rets

        adv_b = advs
        std, mean = adv_b.std(dim=0), adv_b.mean()
        adv_b = (adv_b - mean) / (std + epsilon)

        pi_opt.zero_grad()

        dist = pi.distribution(pi(obs_b), k / config["steps"])
        avg_entropy = dist.entropy().sum() / obs_b.shape[0]
        logp_b = dist.log_prob(a_b)
        assert len(logp_b.shape) == 1
        pi_loss = -(logp_b * adv_b).mean() - eta * avg_entropy
        pi_loss.backward()
        pi_opt.step()

        for i in range(config["vf_train_iters"]):
            vf_opt.zero_grad()
            v_b = vf(obs_b).squeeze()
            vf_loss = ((v_b - ret_b) ** 2).mean()
            vf_loss.backward()
            vf_opt.step()

        if step % config["log_step"] == 0:
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
                    f"avg eps length: {avg_eps_len:.2f}",
                    f"min eps length: {min_eps_len}",
                    f"pi loss: {pi_loss.item():.6f}",
                    f"vf_loss: {vf_loss:.6f}",
                ]
            )
        )

        if np.isclose(avg_entropy.item(), 0.0):
            print("Stopping early: entropy has gone to zero.")
            break
