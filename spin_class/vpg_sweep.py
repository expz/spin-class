import gym
import random
import torch
import wandb

import spin_class.algos.vpg as vpg
import spin_class.config as conf
import spin_class.utils as utils


def main():
    parser = utils.arg_parser()
    parser.add_argument("--seed", default=0, type=int, help="random seed (default 0)")

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
        print(f"Using GPU {i}")
    else:
        device = torch.device(args.device)

    config = conf.default_config[args.env]
    utils.update_config(config, args)
    if args.seed is not None:
        config["seed"] = args.seed

    env = gym.make(config["env"])

    run = wandb.init(project=f"vpg-{args.env}", config=config)

    return vpg.train(env, wandb.config, device, run.id, run.name)


if __name__ == "__main__":
    main()
