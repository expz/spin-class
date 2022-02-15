import gym
import gym.wrappers.time_limit as time_limit
import random
import torch
import wandb

import spin_class.algos.c51 as c51
import spin_class.algos.ddpg as ddpg
import spin_class.algos.ddqn as ddqn
import spin_class.algos.td3 as td3
import spin_class.algos.vpg as vpg
import spin_class.config as conf
import spin_class.utils as utils


def main():
    parser = utils.arg_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        print("WARNING: Unknown arguments:", unknown)

    if args.env not in conf.default_config[args.algo]:
        raise NotImplementedError(
            f"The supplied environment is not currently supported for algorithm {args.algo}:",
            args.env,
        )

    if args.device == "cuda:random":
        i = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{i}")
        print(f"Using GPU {i}")
    else:
        device = torch.device(args.device)

    config = conf.default_config[args.algo][args.env]
    config = utils.add_defaults(args.algo, config)
    utils.update_config(config, args)

    kwargs = config["env_args"] if "env_args" in config else {}
    env = gym.make(config["env"], **kwargs)
    if config["max_episode_steps"] is not None:
        env = time_limit.TimeLimit(
            env.unwrapped, max_episode_steps=config["max_episode_steps"]
        )

    run = wandb.init(project=f"{args.algo}-{args.env}", config=config)

    # So that we can use log_uniform distribution for sweeping, some of these
    # might be floats. Convert them to ints.
    run_config = utils.add_defaults(args.algo, wandb.config)
    for key in [
        "trunk_num_layers",
        "trunk_layer_size",
        "vf_num_layers",
        "vf_layer_size",
        "pi_num_layers",
        "pi_layer_size",
    ]:
        run_config[key] = int(run_config[key])
    if args.algo == "vpg":
        return vpg.train(env, run_config, device, run.id, run.name)
    elif args.algo == "ddqn":
        return ddqn.train(env, run_config, device, run.id, run.name)
    elif args.algo == "c51":
        return c51.train(env, run_config, device, run.id, run.name)
    elif args.algo == "ddpg":
        return ddpg.train(env, run_config, device, run.id, run.name)
    elif args.algo == "td3":
        return td3.train(env, run_config, device, run.id, run.name)


if __name__ == "__main__":
    main()
