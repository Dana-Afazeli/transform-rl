import argparse
import os
import torch

from config import default_config
from src.agents.ddpg import DDPG
from src.envs.mountain_car_continuous_env import MountainCarContinuous
from src.utils.loops import simple_ddpg_loop

def get_args():
    parser = argparse.ArgumentParser()
    for k, v in default_config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))

    return parser.parse_args()

def main():
    args = get_args()

    env = MountainCarContinuous(augment_reward=True)
    _, action_dim = env.get_action_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDPG(
        action_dim,
        args.lr ,
        device,
        args.buffer_size,
        args.batch_size,
        args.weight_decay,
        args.actor_layers,
        args.critic_layers,
        args.encoder_layers,
        args.discount,
        args.tau,
        args.noise_var,
        args.min_noise_var,
        args.noise_var_decay_rate,
        args.random_seed,
    )
    agent.summary()

    simple_ddpg_loop(agent, env, args.episodes, args.max_timesteps)

if __name__ == "__main__":
    main()