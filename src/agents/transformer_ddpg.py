import copy
import torch.nn.functional as F

from agents.base import *
from agents.utils import *
from torch import nn



class TransformerDDPG(BaseAgent):
    def __init__(self, action_dim, actor_lr, critic_lr, device, buffer_size,
            batch_size, weight_decay, actor_config, critic_layers, context_len,
            discount=0.99, tau=0.001, noise_var = 0.001, min_noise_var=0.001, 
            noise_var_decay_rate = 0, random_seed = 42):
        super(BaseAgent, self).__init__()
        self.action_dim = action_dim
        self.reduced_action_dim = self._reduce_dim(action_dim)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.context_len = context_len
        self.discount = discount
        self.tau = tau
        self.random_seed = random_seed
        self.noise_var = noise_var
        self.max_noise_var = noise_var
        self.min_noise_var = min_noise_var
        self.noise_var_decay_rate = noise_var_decay_rate

        self.actor = TransfomerNetwork(actor_config)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay
        )

        self.critic = VanillaCritic(critic_layers).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

        self.replay_buffer = ContextualExperienceReplayBuffer(
            self.buffer_size, self.batch_size, self.context_len,
            self.random_seed
        )

        self.statistics = {
            'actor_loss': None,
            'critic_loss': None,
            'noise_var': self.noise_var
        }

    def pi(self, states, eval=False):
        if eval:
            self.actor.eval()

        actor_input = torch.FloatTensor(states).to(self.device)
        raw_action = self.actor(actor_input).detach().cpu().reshape(self.action_dim)
        action = (
            raw_action + 
            (self.noise_var**0.5)*torch.randn(*raw_action.shape)
        )
        self._decay_noise_var()

        return action.numpy()

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) == 0:
            return
        
        self.actor.train()
        state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs = map(
            lambda x: torch.Tensor(x).to(self.device),
            self.replay_buffer.sample(resample_until_size=True)
        )

        last_states = state_seqs[:, -1, :]
        last_actions = action_seqs[:, -1, :]
        last_rewards = reward_seqs[:, -1, :]
        last_next_states = next_state_seqs[:, -1, :]
        last_dones = done_seqs[:, -1, :]
        # Note that actor should be able to use any context len
        target_Q = self.critic_target(
            last_next_states, 
            self.actor_target(last_next_states)
        ).detach()
        
        target_Q = last_rewards + ((1.-last_dones) * self.discount * target_Q)

        # Get the current Q-value estimate
        current_Q = self.critic(last_states, last_actions)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Now for the hard fucking part
        actor_loss = -self.critic(
            state_seqs, 
            self.actor(state_seqs, all_actions=True)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._update_statistics(
            actor_loss=actor_loss.detach().cpu().numpy(), 
            critic_loss=critic_loss.detach().cpu().numpy()
        )
        
        return actor_loss, critic_loss
    
    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def summary(self):
        self.actor.summary()
        self.critic.summary()

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