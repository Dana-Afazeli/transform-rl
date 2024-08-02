import numpy as np
import torch

from agents.base import BaseReplayBuffer
from collections import namedtuple
from torchsummary import summary

def get_layers(layers):
    last_dim = layers[0]
    nn_layers = []
    for l in layers[1:]:
        if isinstance(l, int):
            nn_layers.append(
                torch.nn.Linear(last_dim, l)
            )
            last_dim = l
        elif isinstance(l, str):
            if l == 'ReLU':
                nn_layers.append(torch.nn.ReLU())
            elif l == 'tanh':
                nn_layers.append(torch.nn.Tanh())
            elif l == 'BN':
                nn_layers.append(torch.nn.BatchNorm1d(last_dim))
            else:
                # Implement if needed
                pass

    return nn_layers

class MLP(torch.nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList(get_layers(layers))
    
    def forward(self, x):
        result = x
        for l in self.layers:
            result = l(result)
        
        return result

class VanillaActor(torch.nn.Module):
    def __init__(self, state_encoder, layers):
        super(VanillaActor, self).__init__()
        self.state_encoder = state_encoder
        self.layers = torch.nn.ModuleList(get_layers(layers))
    
    def forward(self, state):
        result = self.state_encoder(state)
        for l in self.layers:
            result = l(result)
        
        return result
    
    def summary(self):
        print('Actor has {} Params'.format(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
        )
        # summary(self, (self.state_encoder.layers_dim[0],))


class VanillaCritic(torch.nn.Module):
    def __init__(self, state_encoder, layers):
        super(VanillaCritic, self).__init__()
        self.state_encoder = state_encoder
        self.layers = torch.nn.ModuleList(get_layers(layers))


    def forward(self, state, action):
        enc_state = self.state_encoder(state)
        result = torch.cat((enc_state, action), dim=1)
        for l in self.layers:
            result = l(result)

        return result
    
    def summary(self):
        print('Critic has {} Params'.format(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
        )


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
            self.memory.append(e)
        else:
            self.memory[self.current_index] = e
            self.current_index = (self.current_index + 1) % self.buffer_size

    def sample(self, size=None, resample_until_size=False):
        if size is None:
            size = self.batch_size

        if not resample_until_size:
            size = min(size, len(self))

        indexes = self.rng.integers(low=0, high=len(self), size=size)
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
