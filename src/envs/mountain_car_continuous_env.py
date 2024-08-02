import gymnasium as gym

from envs.base import *

class MountainCarContinuous(BaseEnv):
    def __init__(self, augment_reward=False):
        super(MountainCarContinuous, self).__init__()
        self.env_core = gym.make('MountainCarContinuous-v0')
        self.augment_reward = augment_reward

    def reset(self):
        state, _ = self.env_core.reset()
        return state
    
    def step(self, action):
        ns, r, d, trunc, _ = self.env_core.step(action)
        if self.augment_reward:
            r += ns[0]
        return ns, r, d or trunc

    def get_action_info(self):
        return 'continuous', (1,)

    def get_state_info(self):
        return 'continuous', (2,)
