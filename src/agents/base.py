import numpy as np
import torch

from abc import ABC, abstractmethod
from collections import deque, namedtuple
from torchsummary import summary

class BaseAgent(ABC):

    @abstractmethod
    def pi(self, state):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def summary(self):
        pass


class BaseReplayBuffer(ABC):

    @abstractmethod
    def add(self, experience):
        pass

    @abstractmethod
    def sample(self, size=1):
        pass

    @abstractmethod
    def set_random_seed(self, random_seed):
        pass

    @abstractmethod
    def __len__(self):
        pass


class BaseLinearNetwork(torch.nn.Module, ABC):
    
    def __init__(self, layers_dim, activation='ReLU'):
        self.layers_dim = layers_dim
        self.layers = []
        self.activation = activation
        self._last_layer = None
        self._build_network()

    def _build_network(self):
        for i in range(len(self.layers_dim)-1):
            self.layers.append(
                torch.nn.Linear(self.layers_dim[i], self.layers_dim[i+1])
            )
            if self.activation == 'ReLU':
                self.layers.append(torch.nn.ReLU())
            else:
                #TODO: implement if needed
                pass

    @abstractmethod
    def _forward_last_layer(self, x):
        pass

    def summary(self):
        summary(self, self.layers_dim[0])

    def forward(self, x):
        result = x
        for l in self.layers:
            result = l(result)
        
        return self._last_layer(result)

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
