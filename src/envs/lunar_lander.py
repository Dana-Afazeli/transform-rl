import gym as gym

from envs.base import *

class LunarLander(BaseEnv):
    def __init__(self, continuous=True, render_every=None):
        super(LunarLander, self).__init__()
        self.render_env = gym.make('LunarLander-v2', continuous=continuous, render_mode='human')
        self.core_env = gym.make('LunarLander-v2', continuous=continuous)

        self.render_every = render_every
        self.current_env = None
        self.reset_count = 0

    def reset(self):
        if (
            isinstance(self.render_every, int) and
            (self.reset_count % self.render_every == 0)
        ):
            self.current_env = self.render_env
        else:
            self.current_env = self.core_env

        self.reset_count = self.reset_count + 1
        state, _ = self.current_env.reset()
        return state
    
    def step(self, action):
        ns, r, d, trunc, _ = self.current_env.step(action)

        return ns, r, d or trunc

    def get_action_info(self):
        return 'continuous', (2,)

    def get_state_info(self):
        return 'continuous', (8,)

    def get_random_action(self):
        return self.current_env.action_space.sample()