default_config = {
    'random_seed': 42,
    'buffer_size': 10_000,
    'batch_size': 128,
    'discount': 0.99,
    'tau': 1e-3,
    'actor_lr': 1e-4,
    'critic_lr': 1e-4,
    'weight_decay': 1e-5,
    'episodes': 1000,
    'max_timesteps': 300,
    'encoder_layers': [8],
    'actor_layers': [8, 
        100, 'ReLU', 'BN',
        100, 'ReLU', 'BN',
        2,],
    'critic_layers': [
        8 + 2, 
        100, 'ReLU', 'BN',
        100, 'ReLU', 'BN',
        1],
    'noise_var': 1,
    'min_noise_var': 0.0,
    'noise_var_decay_rate': 2e-4,
    'reset_noise_var_every_n_episodes': 100,
    'render_every': 1,
    'n_random_episodes': 0,
    'save_path': 'results/lunar_lander_ddpg'
}