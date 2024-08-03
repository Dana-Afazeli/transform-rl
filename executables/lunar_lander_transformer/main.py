import argparse
import os
import torch

from config import default_config
from src.agents.transformer_ddpg import TransformerDDPG
from src.envs.lunar_lander import LunarLander
from src.utils import transformer_ddpg_loop_with_recordings, Plotter, Recorder

def get_args():
    parser = argparse.ArgumentParser()
    for k, v in default_config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))

    return parser.parse_args()

def main():
    args = get_args()

    env = LunarLander(continuous=True, render_every=args.render_every)
    _, action_dim = env.get_action_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent_kwargs = {
        'action_dim': action_dim,
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
        'device': device,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'actor_config': args.actor_config,
        'critic_layers': args.critic_layers,
        'context_len': args.context_len,
        'discount': args.discount,
        'tau': args.tau,
        'noise_var': args.noise_var,
        'min_noise_var': args.min_noise_var,
        'noise_var_decay_rate': args.noise_var_decay_rate,
        'random_seed': args.random_seed,
    }
    agent = TransformerDDPG(**agent_kwargs)
        
    agent.summary(str(device))

    recorder = Recorder()
    plotter = Plotter()
    if args.save_path is not None:
        plot_save_path = os.path.join(args.save_path, str(args.random_seed))
    else:
        plot_save_path = None

    transformer_ddpg_loop_with_recordings(
        agent, env, args.episodes, args.max_timesteps, args.n_random_episodes,
        args.reset_noise_var_every_n_episodes, recorder, plotter, plot_save_path

    )

if __name__ == "__main__":
    main()