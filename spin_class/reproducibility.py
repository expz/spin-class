import gym
import gym.wrappers.time_limit as time_limit
import json
import random
import torch
import wandb

import spin_class.algos.c51 as c51
import spin_class.algos.ddpg as ddpg
import spin_class.algos.ddqn as ddqn
import spin_class.algos.ppo as ppo
import spin_class.algos.td3 as td3
import spin_class.algos.vpg as vpg
import spin_class.utils as utils
import spin_class.config as conf


def main():
    parser = utils.arg_parser()
    parser.add_argument(
        "--num-seeds", default=5, type=int, help="number of seeds to try"
    )

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
        print("Using GPU", i)
    else:
        device = torch.device(args.device)

    config = conf.default_config[args.algo][args.env]
    config = utils.add_defaults(args.algo, config)
    utils.update_config(config, args)
    print(json.dumps(config, indent=2))

    kwargs = config["env_args"] if "env_args" in config else {}
    env = gym.make(config["env"], **kwargs)
    if config["max_episode_steps"] is not None:
        env = time_limit.TimeLimit(
            env.unwrapped, max_episode_steps=config["max_episode_steps"]
        )

    start_seed = config["seed"]
    for seed in range(start_seed, start_seed + args.num_seeds):
        print("Using seed", seed)
        config["seed"] = seed

        run = wandb.init(
            project=f"{args.algo}-{args.env}",
            config=config,
            name=f"reproducibility-{seed}",
        )

        if args.algo == "vpg":
            vpg.train(env, config, device, run.id, run.name)
        elif args.algo == "ddqn":
            ddqn.train(env, config, device, run.id, run.name)
        elif args.algo == "c51":
            c51.train(env, config, device, run.id, run.name)
        elif args.algo == "ddpg":
            ddpg.train(env, config, device, run.id, run.name)
        elif args.algo == "td3":
            td3.train(env, config, device, run.id, run.name)
        elif args.algo == "ppo":
            ppo.train(env, config, device, run.id, run.name)

        wandb.finish()


if __name__ == "__main__":
    main()
