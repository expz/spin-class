default_config = {
    "vpg": {
        "defaults": {
            "env": "CartPole-v0",
            "env_args": {},
            "save_max_eps": False,
            "save_final": False,
            "trunk_shared": False,
            "trunk_embed": False,
            "trunk_embed_size": 0,
            "trunk_num_layers": 0,
            "trunk_layer_size": 256,
            "trunk_activation": "relu",
            "vf_embed": False,
            "vf_embed_size": 0,
            "vf_num_layers": 0,
            "vf_layer_size": 256,
            "vf_activation": "relu",
            "pi_embed": False,
            "pi_embed_size": 0,
            "pi_num_layers": 0,
            "pi_layer_size": 256,
            "pi_activation": "relu",
            "pi_lr": 0.003,
            "vf_lr": 0.0015,
            "vf_train_iters": 80,
            "std_logits": -0.5,
            "gamma": 0.995,
            "lambda": 0.99,
            "entropy_eta": 0.0,
            "batch_size": 1024,
            "steps": 163840,
            "log_step": 8192,
            "max_episode_steps": None,
            "seed": 42,
        },
        "cartpole": {
            "env": "CartPole-v0",
            "vf_num_layers": 4,
            "vf_layer_size": 16,
            "vf_activation": "relu",
            "pi_num_layers": 3,
            "pi_layer_size": 128,
            "pi_activation": "relu",
            "pi_lr": 0.003,
            "vf_lr": 0.0015,
            "vf_train_iters": 80,
            "gamma": 0.995,
            "lambda": 0.95,
            "batch_size": 1024,
            "steps": 163840,
            "log_step": 8192,
        },
        "invertedpendulum": {
            "env": "InvertedPendulum-v2",
            "vf_num_layers": 2,
            "vf_layer_size": 128,
            "vf_activation": "relu",
            "pi_num_layers": 3,
            "pi_layer_size": 64,
            "pi_activation": "relu",
            "pi_lr": 0.004,
            "vf_lr": 0.1,
            "vf_train_iters": 320,
            "std_logits": -0.5,
            "gamma": 0.995,
            "lambda": 0.99,
            "batch_size": 1024,
            "steps": 163840,
            "log_step": 8192,
        },
        "frozenlake-nonslippery": {
            "env": "FrozenLake-v1",
            "env_args": {
                "is_slippery": False,
            },
            "trunk_shared": False,
            "vf_embed": True,
            "vf_embed_size": 8,
            "vf_num_layers": 4,
            "vf_layer_size": 128,
            "vf_activation": "relu",
            "pi_embed": True,
            "pi_embed_size": 8,
            "pi_num_layers": 3,
            "pi_layer_size": 256,
            "pi_activation": "relu",
            "pi_lr": 0.0022,
            "vf_lr": 0.004,
            "vf_train_iters": 80,
            "gamma": 0.99,
            "lambda": 0.97,
            "batch_size": 2048,
            "entropy_eta": 0.02,
            "steps": 163840,
            "log_step": 4096,
        },
        "frozenlake": {
            "env": "FrozenLake-v1",
            "env_args": {
                "is_slippery": True,
            },
            "trunk_shared": True,
            "trunk_embed": True,
            "trunk_embed_size": 8,
            "trunk_num_layers": 2,
            "trunk_layer_size": 128,
            "trunk_activation": "relu",
            "vf_num_layers": 2,
            "vf_layer_size": 128,
            "vf_activation": "relu",
            "pi_num_layers": 2,
            "pi_layer_size": 192,
            "pi_activation": "relu",
            "pi_lr": 0.006,
            "vf_lr": 0.00017,
            "vf_train_iters": 120,
            "gamma": 0.9987,
            "lambda": 0.97,
            "batch_size": 1024,
            "entropy_eta": 0.037,
            "steps": 327680,
            "log_step": 4096,
        },
        "halfcheetah": {
            "env": "HalfCheetah-v2",
            "vf_num_layers": 2,
            "vf_layer_size": 64,
            "vf_activation": "tanh",
            "pi_num_layers": 2,
            "pi_layer_size": 64,
            "pi_activation": "tanh",
            "pi_lr": 0.004,
            "vf_lr": 0.004,
            "vf_train_iters": 300,
            "std_logits": -0.5,
            "gamma": 0.97,
            "lambda": 0.97,
            "batch_size": 8000,
            "entropy_eta": 0.0,
            "steps": 800000,
            "log_step": 8000,
            "save_final": True,
            "max_episode_steps": 200,
        },
    },
    "ddqn": {
        "defaults": {
            "env": "CartPole-v0",
            "env_args": {},
            "save_max_eps": False,
            "save_final": False,
            "buffer_size": 100000,
            "buffer_type": "uniform",
            "alpha": 0.6,
            "gamma": 0.995,
            "lr": 0.001,
            "num_layers": 3,
            "layer_size": 64,
            "activation": "relu",
            "embed_size": 0,
            "embed": False,
            "batch_size": 1024,
            "steps": 163840,
            "log_step": 8192,
            "eps_sched_len": 100000,
            "learning_starts": 2048,
            "training_freq": 256,
            "target_update_freq": 8192,
            "max_episode_steps": None,
            "seed": 42,
        },
        "cartpole": {
            "env": "CartPole-v0",
            "save_final": True,
            "num_layers": 3,
            "layer_size": 64,
            "activation": "relu",
            "buffer_size": 300000,
            "buffer_type": "prioritized",
            "lr": 0.004,
            "alpha": 0.6,
            "gamma": 0.995,
            "batch_size": 1024,
            "steps": 327680,
            "log_step": 8192,
            "eps_sched_len": 200000,
            "learning_starts": 2048,
            "training_freq": 64,
            "target_update_freq": 8192,
        },
    },
}
