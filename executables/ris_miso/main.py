import argparse
import os
import torch

from config import default_config
from src.agents.ddpg import DDPG
from src.envs.ris_miso_env import RIS_MISO
from src.utils.loops import simple_ddpg_loop

def get_args():
    parser = argparse.ArgumentParser()
    for k, v in default_config.items():
        parser.add_argument(f'--{k}', default=v)

    return parser.parse_args()

def main():
    args = get_args()

    env = RIS_MISO(
        args.num_antennas,
        args.num_RIS_elements,
        args.num_users,
        args.power_t,
        AWGN_var=args.awgn_var,
    )
    _, state_dim = env.get_state_info()
    _, action_dim = env.get_action_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DDPG(
        state_dim,
        action_dim,
        50,
        1e-3,
        device,
        args.buffer_size,
        args.batch_size,
        args.weight_decay,
        args.discount,
        args.tau,
        args.random_seed,
        activation='tanh'
    )
    agent.summary()

    simple_ddpg_loop(agent, env, args.episodes, args.max_timesteps)

if __name__ == "__main__":
    main()