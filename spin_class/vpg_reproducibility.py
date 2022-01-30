import gym
import json
import random
import spin_class.algos.vpg as vpg
import torch
import wandb

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

    if args.env not in conf.default_config:
        raise NotImplementedError(
            "The supplied environment is not currently supported:", args.env
        )

    if args.device == "cuda:random":
        i = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{i}")
        print("Using GPU", i)
    else:
        device = torch.device(args.device)

    config = conf.default_config[args.env]
    utils.update_config(config, args)
    print(json.dumps(config, indent=2))

    kwargs = config["env_args"] if "env_args" in config else {}
    env = gym.make(config["env"], **kwargs)

    start_seed = config["seed"]
    for seed in range(start_seed, start_seed + args.num_seeds):
        print("Using seed", seed)
        config["seed"] = seed

        run = wandb.init(
            project=f"vpg-{args.env}",
            config=config,
            name=f"reproducibility-{seed}",
        )

        vpg.train(env, config, device, run.id, run.name)

        wandb.finish()


if __name__ == "__main__":
    main()
