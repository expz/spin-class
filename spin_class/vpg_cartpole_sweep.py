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
    "pi_lr": 0.01,
    "vf_lr": 0.01,
    "vf_train_iters": 80,
    "gamma": 0.995,
    "lambda": 0.95,
    "batch_size": 8192,
    "steps": 163840,
    "log_step": 8192,
    "seed": 42,
    "method": "bayes",
}


def main():
    run = wandb.init(config=default_config)

    i = random.randrange(torch.cuda.device_count())
    device = torch.device(f"cuda:{i}")

    env = gym.make("CartPole-v0")

    return vpg.train(env, wandb.config, device, run.id, run.name)


if __name__ == "__main__":
    main()
