import numpy as np
import torch

from base import BaseReplayBuffer, BaseLinearNetwork
from collections import namedtuple

class VanillaStateEncoder(BaseLinearNetwork):
    def _forward_last_layer(self, x):
        return x

class VanillaActor(BaseLinearNetwork):
    def __init__(self, state_encoder, layers_dim, activation='ReLU'):
        super(VanillaActor, self).__init__(layers_dim, activation)
        self.state_encoder = state_encoder

    def _forward_last_layer(self, x):
        return x
    
    def forward(self, state):
        enc_state = self.state_encoder(state)
        return super().forward(enc_state)

class VanillaCritic(BaseLinearNetwork):
    def __init__(self, state_encoder, layers_dim, activation='ReLU'):
        super(VanillaCritic, self).__init__(layers_dim, activation)
        self.state_encoder = state_encoder

    def _forward_last_layer(self, x):
        return x
    
    def forward(self, state, action):
        enc_state = self.state_encoder(state)
        state_action = torch.cat((enc_state, action))
        return super().forward(state_action)


class VanillaExperienceReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer_size, batch_size, random_seed=42):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.reset()
        self.experience = namedtuple(
            "Experience", 
            field_names=["state", "action", "reward", "next_state", "done"]
        )

    def reset(self):
        self.memory = []
        self.current_index = 0
        self.rng = np.random.default_rng(self.random_seed)
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        if len(self) < self.buffer_size:
            self.memory[self.current_index].append(e)
        else:
            self.memory[self.current_index] = e
            self.current_index += (self.current_index + 1) % self.buffer_size

    def sample(self):
        indexes = self.rng.integers(low=0, high=len(self), size=self.batch_size)

        states = np.vstack([self.memory[idx].state for idx in indexes])
        actions = np.vstack([self.memory[idx].action for idx in indexes])
        rewards = np.vstack([self.memory[idx].reward for idx in indexes])
        next_states = np.vstack([self.memory[idx].next_state for idx in indexes])
        dones = np.vstack([self.memory[idx].done for idx in indexes]).astype(np.float16)

        return (states, actions, rewards, next_states, dones)

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def __len__(self):
        return len(self.memory)
