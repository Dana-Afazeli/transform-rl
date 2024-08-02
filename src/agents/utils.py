import numpy as np
import torch

from agents.base import BaseReplayBuffer
from collections import namedtuple
from torchsummary import summary
from torch import nn

def get_layers(layers):
    last_dim = layers[0]
    nn_layers = []
    for l in layers[1:]:
        if isinstance(l, int):
            layer = torch.nn.Linear(last_dim, l)
            torch.nn.init.xavier_uniform_(layer.weight)
            nn_layers.append(
                layer
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

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.input_dim = layers[0]
        self.layers = torch.nn.ModuleList(get_layers(layers))
    
    def forward(self, x):
        result = x
        for l in self.layers:
            result = l(result)
        
        return result

    def summary(self):
        summary(self, (self.input_dim,))

class VanillaCritic(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.input_dim = layers[0]
        self.layers = torch.nn.ModuleList(get_layers(layers))
    
    def forward(self, x, y=None):
        if y is None:
            result = x
        else:
            dim = len(x.shape) - 1
            result = torch.stack([x, y], dim=dim)

        for l in self.layers:
            result = l(result)
        
        return result

    def summary(self):
        summary(self, (self.input_dim,))

class TransfomerNetwork(nn.Module):
    def __init__(self, config):
        super(TransfomerNetwork, self).__init__()
        self.input_dim = config['input_layers'][0]
        self.context_len = config['context_len']
        self.networks = self._init_networks(config)
        self.params = torch.nn.ModuleList(self.networks.values())
    
    def _init_networks(self, config):
        netowrks = {
            'input_layer': MLP(*config['input_layers']),
            'output_layer': MLP(*config['output_layers']),
            'positional_embedding': nn.Parameter(
                torch.randn(config['context_len'], config['input_layer'][1])/1e3
            ),
            'decoder': nn.TransformerDecoderLayer(
                d_model=config['output_layers'][0],
                nhead=4,
                dim_feedforward=config['output_layers'][0],
                dropout=0.1, 
                batch_first=True
            )
        }

    def _generate_mask(self, context_size):
        mask = (
            torch.triu(torch.ones(context_size, context_size)) == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def forward(self, x, all_actions=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # x: (batch_size, context_size, state_size)
        enc_x = self.networks['input_layer'](x)
        enc_x += (
            self.networks['positional_embedding']
            .unsqueeze(0).repeat(enc_x.shape[0], 1, 1)
            [:, :enc_x.shape[1], :]
        )

        mask = self._generate_mask(enc_x.shape[1])
        decoder_output = (
            self.networks['decoder']
            (enc_x, enc_x, tgt_mask=mask)
        )

        actions = self.networks['output_layer'](decoder_output)
        if all_actions:
            return actions
        else:
            return actions[:, -1, :]
    
    def summary(self):
        summary(self, (1, self.context_len, self.input_dim,))
    
class ContextualExperienceReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer_size, batch_size, context_size, random_seed=42):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.context_size = context_size
        self.random_seed = random_seed
        self.reset()
        self.experience = namedtuple(
            'Experience',
            field_names=['state', 'action', 'next_state', 'reward', 'done']
        )
        self.experience_sequence = namedtuple(
            'ExperienceSequence',
            field_names=[
                'state_sequence',
                'action_sequence',
                'reward_sequence',
                'done_sequence'
            ]
        )

    def add(self, state, action, reward, next_state, done):
        self.stm['states'].append(state)
        self.stm['states'] = self.stm['states'][-self.context_size:]
        self.stm['actions'].append(action)
        self.stm['actions'] = self.stm['action'][-self.context_size:]
        self.stm['rewards'].append(reward)
        self.stm['rewards'] = self.stm['reward'][-self.context_size:]
        self.stm['next_states'].append(next_state)
        self.stm['next_states'] = self.stm['next_state'][-self.context_size:]
        self.stm['dones'].append(done)
        self.stm['dones'] = self.stm['done'][-self.context_size:]

        if len(self.stm['states']) == self.context_size:
            self._add_sequence()

    def _add_sequence(self):
        e = self.experience_sequence(
            self.stm['states'],
            self.stm['actions'],
            self.stm['rewards'],
            self.stm['dones']
        )
        if len(self) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.current_index] = e
            self.current_index = (self.current_index + 1) % self.buffer_size


    def sample(self, size=1, resample_until_batch_size=False):
        if size is None:
            size = self.batch_size

        if not resample_until_batch_size:
            size = min(size, len(self))

        indexes = self.rng.integers(low=0, high=len(self), size=size)
        states = np.vstack(
            [np.vstack(self.memory[idx].states) for idx in indexes]
        )
        actions = np.vstack(
            [np.vstack(self.memory[idx].actions) for idx in indexes]
        )
        rewards = np.vstack(
            [np.vstack(self.memory[idx].rewards) for idx in indexes]
        )
        next_states = np.vstack(
            [np.vstack(self.memory[idx].next_states) for idx in indexes]
        )
        dones = np.vstack(
            [np.vstack(self.memory[idx].dones) for idx in indexes]
        )

        return (states, actions, rewards, next_states, dones)

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def reset(self):
        self.memory = []
        self.stm = {
            'states':[],
            'actions':[],
            'next_states':[],
            'rewards':[],
            'dones':[]
        }
        self.current_index = 0
        self.rng = np.random.default_rng(self.random_seed)

    def __len__(self):
        return len(self.memory)

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