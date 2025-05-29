import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, activation=nn.Tanh(), device='cpu', writer=None):
        super(ActorCritic, self).__init__()
        self.device = device
        self.writer = writer
        
        # Body network
        self.body = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            activation,
            nn.Linear(n_latent_var, n_latent_var),
            activation
        ).to(self.device)
        
        # Actor head
        self.action_layer = nn.Sequential(
            self.body,
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Critic head
        self.value_layer = nn.Sequential(
            self.body,
            nn.Linear(n_latent_var, 1)
        ).to(self.device)

    def forward(self, state):   # For the onnx export only
        #state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        return action_probs

    def act(self, state, memory, permute=False):
        # Receive numpy array
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        
        if permute:
            # Create permutation tensor on the same device as action_probs
            perm = torch.tensor([2, 4, 3, 1, 0], device=self.device)
            # Use torch.index_select for efficient permutation
            action_probs = action_probs.index_select(1, perm)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        # Return numpy array
        return action.cpu().numpy()

    def evaluate(self, state, action):
        state = state.to(self.device)
        
        
        # Get intermediate values
        body_output = self.body(state)
        print("Body output mean:", body_output.mean().item())
        print("Body output std:", body_output.std().item())
        
        action_probs = self.action_layer(state)
        print("Action probs mean:", action_probs.mean().item())
        print("Action probs std:", action_probs.std().item())
        if self.writer:
            self.writer.add_scalar('action_probs/mean', action_probs.mean().item())
            self.writer.add_scalar('action_probs/std', action_probs.std().item())
        
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state.to(self.device))
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy