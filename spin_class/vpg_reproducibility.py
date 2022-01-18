import argparse
import gym
import random
import spin_class.algos.vpg as vpg
import torch
import wandb

from typing import Any, Dict

default_config = {
    "cartpole": {
        "env": "CartPole-v0",
        "vf_layers": [
            ("linear", 16, "relu"),
            ("linear", 16, "relu"),
            ("linear", 16, "relu"),
            ("linear", 16, "relu"),
        ],
        "pi_layers": [
            ("linear", 128, "relu"),
            ("linear", 128, "relu"),
            ("linear", 128, "relu"),
        ],
        "pi_lr": 0.003,
        "vf_lr": 0.0015,
        "vf_train_iters": 80,
        "gamma": 0.995,
        "lambda": 0.95,
        "batch_size": 1024,
        "steps": 163840,
        "log_step": 8192,
        "seed": 42,
    },
    "invertedpendulum": {
        "env": "InvertedPendulum-v2",
        "vf_layers": [
            ("linear", 128, "relu"),
            ("linear", 128, "relu"),
        ],
        "pi_layers": [
            ("linear", 64, "relu"),
            ("linear", 64, "relu"),
            ("linear", 64, "relu"),
        ],
        "pi_lr": 0.004,
        "vf_lr": 0.1,
        "vf_train_iters": 320,
        "std_logits": -0.5,
        "gamma": 0.995,
        "lambda": 0.99,
        "batch_size": 1024,
        "steps": 163840,
        "log_step": 8192,
        "seed": 42,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="check how dependent the performance of a model is on the random seed"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="environment to use: cartpole, invertedpendulum, frozenlake, halfcheetah",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device on which to run training algorithm: cpu, cuda:random, cuda:0, ...",
    )
    parser.add_argument(
        "--logstep",
        default=None,
        type=int,
        help="log statistics every time this many steps are taken",
    )
    parser.add_argument(
        "--stdlogits",
        default=None,
        type=float,
        help="for continuous control, the standard deviation of the Gaussian in logits",
    )
    parser.add_argument(
        "--vftrainiters",
        default=None,
        type=int,
        help="value function training iterations",
    )
    parser.add_argument(
        "--lam",
        default=None,
        type=float,
        help="lambda for generalized advantage estimation",
    )
    parser.add_argument(
        "--steps", default=None, type=int, help="number of steps to train"
    )
    parser.add_argument(
        "--num-seeds", default=5, type=int, help="number of seeds to try"
    )
    parser.add_argument(
        "--vflr", default=None, type=float, help="value function learning rate"
    )
    parser.add_argument("--pilr", default=None, type=float, help="policy learning rate")
    parser.add_argument("--gamma", default=None, type=float, help="discount factor")
    parser.add_argument("--batchsize", default=None, type=int, help="batch size")
    args, unknown = parser.parse_known_args()

    if unknown is not None:
        print("Unknown arguments:", unknown)

    return args


def update_config(config: Dict[str, Any], args: argparse.Namespace):
    if args.batchsize is not None:
        config["batch_size"] = args.batchsize
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.lam is not None:
        config["lambda"] = args.lam
    if args.logstep is not None:
        config["log_step"] = args.logstep
    if args.pilr is not None:
        config["pi_lr"] = args.pilr
    if args.stdlogits is not None:
        config["std_logits"] = args.stdlogits
    if args.steps is not None:
        config["steps"] = args.steps
    if args.vflr is not None:
        config["vf_lr"] = args.vflr
    if args.vftrainiters is not None:
        config["vf_train_iters"] = args.vftrainiters


def main():
    args = parse_args()

    if args.env not in default_config:
        raise NotImplementedError(
            "The supplied environment is not currently supported:", args.env
        )

    if args.device == "cuda:random":
        i = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{i}")
        print("Using GPU", i)
    else:
        device = torch.device(args.device)

    config = default_config[args.env]
    update_config(config, args)

    env = gym.make(config["env"])

    for seed in range(args.num_seeds):
        config["seed"] = seed

        run = wandb.init(
            project=f"vpg-{args.env}",
            config=config,
            name=f"reproducibility-{seed}",
        )

        vpg.train(env, config, device, run.id, run.name)


if __name__ == "__main__":
    main()
