import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class BaseEnv(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_action_info(self):
        pass

    @abstractmethod
    def get_state_info(self):
        pass
