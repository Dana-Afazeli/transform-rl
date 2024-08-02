import copy
import torch
import torch.nn.functional as F

from agents.base import *
from functools import reduce
from agents.utils import *

class DDPG(BaseAgent):
    def __init__(
            self, action_dim, actor_lr, critic_lr, device, buffer_size,
            batch_size, weight_decay, actor_layers, critic_layers, encoder_layers,
            discount=0.99, tau=0.001, noise_var = 0.001, min_noise_var=0.001, 
            noise_var_decay_rate = 0, random_seed = 42
        ):
        super(BaseAgent, self).__init__()
        self.action_dim = action_dim
        self.reduced_action_dim = self._reduce_dim(self.action_dim)
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

        self.state_encoder = MLP(encoder_layers).to(device)

        self.actor = VanillaActor(
            self.state_encoder, 
            actor_layers
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay
        )


        self.critic = VanillaCritic(
            self.state_encoder, 
            critic_layers
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

        self.replay_buffer = VanillaExperienceReplayBuffer(
            self.buffer_size, 
            self.batch_size,
            self.random_seed
        )
        
    def summary(self):
        self.actor.summary()
        self.critic.summary()

    def pi(self, state, eval=False):
        if eval:
            self.actor.eval()

        actor_input = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        raw_action = self.actor(actor_input).detach().cpu().reshape(self.action_dim)
        action = (
            raw_action + 
            (self.noise_var**0.5)*torch.randn(*raw_action.shape)
        )
        self._decay_noise_var()

        return action.numpy()

    def step(self, state, action, reward, next_state, done):
        self.actor.train()
        self.replay_buffer.add(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = map(
            lambda x: torch.Tensor(x).to(self.device),
            self.replay_buffer.sample(resample_until_size=True)
        )
        target_Q = self.critic_target(
            next_states, 
            self.actor_target(next_states)
        )
        target_Q = rewards + ((1.-dones) * self.discount * target_Q).detach()

        # Get the current Q-value estimate
        current_Q = self.critic(states, actions)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss

    # Save the model parameters
    def save(self, filepath):
        torch.save(self.critic.state_dict(), filepath + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filepath + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filepath + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filepath + "_actor_optimizer")

    # Load the model parameters
    def load(self, filepath):
        self.critic.load_state_dict(torch.load(filepath + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filepath + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filepath + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filepath + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

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

    def reset_noise_var(self):
        self.noise_var = self.max_noise_var