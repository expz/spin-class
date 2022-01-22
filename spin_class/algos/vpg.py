from datetime import datetime
import gym
import gym.spaces
import numpy as np
import os
import random
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any, Dict, List, Tuple, Union
import wandb


def shape(space: gym.Space):
    if isinstance(space, gym.spaces.Discrete):
        return tuple()
    elif isinstance(space, gym.spaces.Box):
        return space.sample().shape
    else:
        raise Exception(f"Unsupported space type: {type(space)}")


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

        if isinstance(env.observation_space, gym.spaces.Box):
            self.layers = [("input", shape(env.observation_space)[0])] + layers
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            self.layers = [
                ("input", 1),
            ] + layers

        self.model = None

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def build_model(self, conf: List[Union[Tuple[str, int], Tuple[str, int, str]]]):
        layers = []
        t, prev_size = conf[0]
        assert t == "input"
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
                    raise NotImplementedError(
                        f"Unrecognized activation type: {layer[2]}"
                    )
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
                        device=self.device,
                    )
                )
            else:
                raise ValueError(f"Unrecognized layer type: {layer[0]}")
            prev_size = size
        return nn.Sequential(*layers).to(self.device)


class VPGValueModel(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.DeviceObjType,
    ):
        super(VPGValueModel, self).__init__(env, layers, device)

        self.layers += [("linear", 1, "none")]
        self.model = self.build_model(self.layers)


class VPGPolicyModel(VPGModel):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.DeviceObjType,
    ):
        super(VPGPolicyModel, self).__init__(env, layers, device)


class VPGGaussianPolicyModel(VPGPolicyModel):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        std_logits: float,
        device: torch.torch.DeviceObjType,
    ):
        super(VPGGaussianPolicyModel, self).__init__(env, layers, device)

        assert isinstance(env.action_space, gym.spaces.Box)

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

        self.layers += output_layers
        self.model = self.build_model(self.layers)

        self.std = torch.exp(
            std_logits * torch.ones((signal_count,), dtype=torch.float32, device=device)
        )

    def distribution(self, output: torch.Tensor):
        return Normal(output, self.std)


class VPGCategoricalPolicyModel(VPGPolicyModel):
    def __init__(
        self,
        env: gym.Env,
        layers: List[Tuple[str, int, str]],
        device: torch.torch.DeviceObjType,
    ):
        super(VPGCategoricalPolicyModel, self).__init__(env, layers, device)

        assert isinstance(env.action_space, gym.spaces.Discrete)

        output_layers = [("linear", env.action_space.n, "none")]
        self.layers += output_layers
        self.model = self.build_model(self.layers)

    def distribution(self, output: torch.Tensor):
        return Categorical(logits=output)


def train(
    env: gym.Env,
    config: Dict[str, Any],
    device: torch.DeviceObjType,
    run_id: str,
    run_name: str,
):

    # Make the training reproducible.
    env.seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if isinstance(env.action_space, gym.spaces.Box):
        pi = VPGGaussianPolicyModel(
            env, config["pi_layers"], config["std_logits"], device
        )
    elif isinstance(env.action_space, gym.spaces.Discrete):
        pi = VPGCategoricalPolicyModel(env, config["pi_layers"], device)
    else:
        raise NotImplementedError(
            f"Action space type not yet supported: {type(env.action_space)}"
        )
    vf = VPGValueModel(env, config["vf_layers"], device)

    wandb.watch(pi)
    wandb.watch(vf)

    pi_opt = Adam(pi.parameters(), lr=config["pi_lr"])
    vf_opt = Adam(vf.parameters(), lr=config["vf_lr"])

    save_max_eps = config["save_max_eps"]
    gamma = config["gamma"]
    lam = config["lambda"]
    batch_size = config["batch_size"]
    avg_eps_len = 0
    max_avg_eps_rew = float("-inf")
    max_performance = False
    epsilon = 1e-6
    obs_dtype = (
        torch.int64
        if isinstance(env.observation_space, gym.spaces.Discrete)
        else torch.float32
    )

    model_dir = f"models/vpg/{env.spec.id.lower()}"
    os.makedirs(f"{model_dir}/pi", mode=0o755, exist_ok=True)
    os.makedirs(f"{model_dir}/vf", mode=0o755, exist_ok=True)

    for k in range(0, config["steps"], batch_size):
        done = False
        obs = env.reset()
        obss = np.zeros([batch_size] + list(shape(env.observation_space)))
        if len(obss.shape) == 1:
            obss = np.expand_dims(obss, axis=1)
        rets = torch.zeros(batch_size, dtype=torch.float32, device=device)
        advs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        as_ = torch.zeros(batch_size, dtype=torch.float32, device=device)
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
                dist = pi.distribution(p)
                a = dist.sample()
                obs, r, done, _ = env.step(a.item())

            eps_vs[eps_len - 1] = v
            as_[i] = a
            eps_rs[eps_len - 1] = r
            rs[i] = r

            total_rew += r

            if done or i == batch_size - 1:
                if done:
                    eps_lens.append(eps_len)
                    total_rews.append(total_rew)
                ret = 0
                for i in range(eps_len - 1, -1, -1):
                    ret = eps_rs[i] + gamma * ret
                    rets[ptr + i] = ret
                adv = 0
                for i in range(eps_len - 1, 0, -1):
                    adv = (
                        eps_rs[i - 1] + gamma * eps_vs[i] - eps_vs[i - 1]
                    ) + lam * gamma * adv
                    advs[ptr + i] = adv
                advs[ptr + eps_len - 1] = eps_rs[-1] - eps_vs[-1]
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
                max_performance = True
            else:
                max_performance = False

        obs_b = torch.as_tensor(obss.squeeze(), dtype=obs_dtype, device=device)
        a_b = as_
        ret_b = rets

        adv_b = advs
        std, mean = adv_b.std(dim=0), adv_b.mean()
        adv_b = (adv_b - mean) / (std + epsilon)

        pi_opt.zero_grad()

        dist = pi.distribution(pi(obs_b))
        avg_entropy = dist.entropy().sum() / obs_b.shape[0]
        logp_b = dist.log_prob(a_b)
        pi_loss = -(logp_b * adv_b).mean()
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
