import gym
import random
import spin_class.algos.vpg as vpg
import torch
import wandb


default_config = {
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
    "method": "bayes",
}

NUM_SEEDS = 10


def main():
    for seed in range(NUM_SEEDS):
        default_config["seed"] = seed
        run = wandb.init(
            project="vpg-cartpole",
            config=default_config,
            name=f"reproducibility-{seed}",
        )

        i = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{i}")

        env = gym.make("CartPole-v0")

        return vpg.train(env, default_config, device, run.id, run.name)


if __name__ == "__main__":
    main()
