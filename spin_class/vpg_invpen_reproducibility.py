import gym
import random
import spin_class.algos.vpg as vpg
import torch
import wandb


default_config = {
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
    "method": "bayes",
}

NUM_SEEDS = 1


def main():
    for seed in range(NUM_SEEDS):
        default_config["seed"] = seed
        run = wandb.init(
            project="vpg-invpen",
            config=default_config,
            name=f"reproducibility-{seed}",
        )

        i = random.randrange(torch.cuda.device_count())
        device = torch.device(f"cuda:{i}")

        env = gym.make(default_config["env"])

        return vpg.train(env, default_config, device, run.id, run.name)


if __name__ == "__main__":
    main()
