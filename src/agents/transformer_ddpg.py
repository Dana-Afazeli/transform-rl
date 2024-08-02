import copy

from agents.base import *
from agents.utils import *
from torch import nn



class TransformerDDPG(BaseAgent):
    def __init__(self, action_dim, actor_lr, critic_lr, device, buffer_size,
            batch_size, weight_decay, actor_layers, critic_layers, context_len,
            discount=0.99, tau=0.001, noise_var = 0.001, min_noise_var=0.001, 
            noise_var_decay_rate = 0, random_seed = 42):
        super(BaseAgent, self).__init__()
        self.action_dim = action_dim
        self.reduced_action_dim = self._reduce_dim(action_dim)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.random_seed = random_seed
        self.noise_var = noise_var
        self.max_noise_var = noise_var
        self.min_noise_var = min_noise_var
        self.noise_var_decay_rate = noise_var_decay_rate

        self.critic = MLP(critic_layers).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

    def pi(self, state, eval=False):
        pass

    def step(self, state, action, reward, next_state, done):
        pass
    
    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def summary(self):
        pass

    def _reduce_dim(self, dim):
        if isinstance(dim, tuple):
            if len(dim) > 1:
                return reduce(lambda x, y: x*y, dim)
            else:
                return dim[0]
        elif isinstance(dim, int):
            return dim
        else:
            raise Exception('wtf is your dim')
        
    def _decay_noise_var(self):
        self.noise_var = max(
            self.noise_var * (1-self.noise_var_decay_rate), 
            self.min_noise_var
        )
        self._update_statistics(noise_var=self.noise_var)

    def reset_noise_var(self):
        self.noise_var = self.max_noise_var
        self._update_statistics(noise_var=self.noise_var)

    def _update_statistics(self, **kwargs):
        for k, v in kwargs.items():
            self.statistics[k] = v

    def report_statistics(self):
        return self.statistics