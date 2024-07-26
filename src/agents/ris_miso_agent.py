import copy
import torch
import torch.nn.functional as F

from base import *
from functools import reduce

class StateEncoder(BaseLinearNetwork):
    def _forward_last_layer(self, x):
        return x

class Actor(BaseLinearNetwork):
    def __init__(self, state_encoder, layers_dim, activation='ReLU'):
        super(Actor, self).__init__(layers_dim, activation)
        self.state_encoder = state_encoder

    def _forward_last_layer(self, x):
        return x
    
    def forward(self, state):
        enc_state = self.state_encoder(state)
        return super().forward(enc_state)

class Critic(BaseLinearNetwork):
    def __init__(self, state_encoder, layers_dim, activation='ReLU'):
        super(Critic, self).__init__(layers_dim, activation)
        self.state_encoder = state_encoder

    def _forward_last_layer(self, x):
        return x
    
    def forward(self, state, action):
        enc_state = self.state_encoder(state)
        state_action = torch.cat((enc_state, action))
        return super().forward(state_action)

class DDPG(BaseAgent):
    def __init__(
            self, state_dim, action_dim, encoding_dim, lr, device,
            weight_decay, discount=0.99, tau=0.001
        ):
        self.state_dim = state_dim
        self.reduced_state_dim = self._reduce_dim(self.state_dim)
        self.action_dim = action_dim
        self.reduced_action_dim = self._reduce_dim(self.action_dim)
        self.device = device
        self.discount = discount
        self.tau = tau

        self.state_encoder = StateEncoder(
            [self.reduced_state_dim, 
             (self.reduced_state_dim+encoding_dim)/2, encoding_dim]
        ).to(self.device)


        self.actor = Actor(
            self.state_encoder, 
            [encoding_dim, 
             (encoding_dim+self.reduced_action_dim)/2, self.reduced_action_dim]
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay
        )


        self.critic = Critic(
            self.state_encoder,
            [encoding_dim + self.reduced_action_dim, 
             (encoding_dim + self.reduced_action_dim)/2, 1]
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )
        
    def summary(self):
        self.actor.summary()
        self.critic.summary()

    def pi(self, state):
        self.actor.eval()

        actor_input = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(actor_input).cpu().numpy().reshape(self.action_dim)

        return action

    def step(self, replay_buffer, batch_size=16):
        self.actor.train()

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get the current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

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