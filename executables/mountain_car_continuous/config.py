default_config = {
    'random_seed': 42,
    'buffer_size': 10_000,
    'batch_size': 128,
    'discount': 0.99,
    'tau': 1e-3,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'episodes': 100,
    'max_timesteps': 10_000,
    'encoder_layers': [2],
    'actor_layers': [2, 2, 'ReLU', 1, 'tanh'],
    'critic_layers': [2 + 1, 3, 'ReLU', 1],
    'noise_var': 1,
    'min_noise_var': 0.3,
    'noise_var_decay_rate': 0.001
}