import torch

from abc import ABC, abstractmethod

from torchsummary import summary

class BaseAgent(ABC):

    @abstractmethod
    def pi(self, state, eval=False):
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

    @abstractmethod
    def report_statistics(self):
        pass


class BaseReplayBuffer(ABC):

    @abstractmethod
    def add(self, experience):
        pass

    @abstractmethod
    def sample(self, size=1, resample_until_batch_size=False):
        pass

    @abstractmethod
    def set_random_seed(self, random_seed):
        pass

    @abstractmethod
    def __len__(self):
        pass