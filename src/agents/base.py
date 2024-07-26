import torch

from abc import ABC, abstractmethod
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
