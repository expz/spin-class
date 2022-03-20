import argparse

from typing import Any, Dict

import spin_class.config as conf


def arg_parser():
    parser = argparse.ArgumentParser(
        description="check how dependent the performance of a model is on the random seed"
    )
    parser.add_argument(
        "--algo",
        default="vpg",
        const="vpg",
        nargs="?",
        choices=["vpg", "ddqn", "c51", "ddpg", "td3", "ppo"],
        help="algorithm (default: %(default)s)",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device on which to run training algorithm: cpu, cuda:random, cuda:0, ...",
    )
    parser.add_argument(
        "--env",
        required=True,
        help="environment to use: cartpole, invertedpendulum, frozenlake, halfcheetah",
    )
    parser.add_argument("--gamma", default=None, type=float, help="discount factor")
    parser.add_argument(
        "--lam",
        default=None,
        type=float,
        help="lambda for generalized advantage estimation",
    )
    parser.add_argument(
        "--log_step",
        default=None,
        type=int,
        help="log statistics every time this many steps are taken",
    )
    parser.add_argument(
        "--pi_activation",
        default=None,
        type=str,
        help="activation of policy network layers",
    )
    parser.add_argument(
        "--pi_layer_size",
        default=None,
        type=float,
        help="number of units in policy network layer",
    )
    parser.add_argument(
        "--pi_lr", default=None, type=float, help="policy learning rate"
    )
    parser.add_argument(
        "--pi_num_layers",
        default=None,
        type=int,
        help="number of layers in policy network",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed (default 0)")
    parser.add_argument(
        "--std_logits",
        default=None,
        type=float,
        help="for continuous control, the standard deviation of the Gaussian in logits",
    )
    parser.add_argument(
        "--steps", default=None, type=int, help="number of steps to train"
    )
    parser.add_argument(
        "--vf_activation",
        default=None,
        type=str,
        help="activation of value function layers",
    )
    parser.add_argument(
        "--vf_layer_size",
        default=None,
        type=float,
        help="number of units in value function layers",
    )
    parser.add_argument(
        "--vf_lr", default=None, type=float, help="value function learning rate"
    )
    parser.add_argument(
        "--vf_num_layers",
        default=None,
        type=int,
        help="number of value function layers",
    )
    parser.add_argument(
        "--vf_train_iters",
        default=None,
        type=int,
        help="value function training iterations",
    )

    return parser


def update_config(config: Dict[str, Any], args: argparse.Namespace):
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.lam is not None:
        config["lambda"] = args.lam
    if args.log_step is not None:
        config["log_step"] = args.log_step
    if args.seed is not None:
        config["seed"] = args.seed
    if args.pi_activation is not None:
        config["pi_activation"] = args.pi_activation
    if args.pi_lr is not None:
        config["pi_lr"] = args.pi_lr
    if args.pi_layer_size is not None:
        config["pi_layer_size"] = int(args.pi_layer_size)
    if args.pi_num_layers is not None:
        config["pi_num_layers"] = args.pi_num_layers
    if args.std_logits is not None:
        config["std_logits"] = args.std_logits
    if args.steps is not None:
        config["steps"] = args.steps
    if args.vf_activation is not None:
        config["vf_activation"] = args.vf_activation
    if args.vf_lr is not None:
        config["vf_lr"] = args.vf_lr
    if args.vf_layer_size is not None:
        config["vf_layer_size"] = int(args.vf_layer_size)
    if args.vf_num_layers is not None:
        config["vf_num_layers"] = args.vf_num_layers
    if args.vf_train_iters is not None:
        config["vf_train_iters"] = args.vf_train_iters


def add_defaults(algo: str, config: Dict[str, Any]):
    defaults = conf.default_config[algo]["defaults"].copy()
    for k, v in config.items():
        defaults[k] = v
    return defaults
