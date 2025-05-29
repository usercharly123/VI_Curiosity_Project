import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FeatureExtractor(nn.Module):
  def __init__(self, hidden_size):
    super(FeatureExtractor, self).__init__()
    
    self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU())
    
    for p in self.modules():
      if isinstance(p, nn.Conv2d):
        init.orthogonal_(p.weight, np.sqrt(2))
        if p.bias is not None:
            init.constant_(p.bias, 0)

      if isinstance(p, nn.Linear):
        init.orthogonal_(p.weight, np.sqrt(2))
        if p.bias is not None:
            nn.init.constant_(p.bias, 0)
  
  def forward(self, states):
    return self.cnn(states)
  

class ICM(nn.Module):
  def __init__(self, hidden_size, n_actions, cnn, device):
    super(ICM, self).__init__()
    self.hidden_size = hidden_size
    self.n_actions = n_actions
    self.device = device

    self.state_features_extractor = cnn
    
    self.inverse_dynamics_model = nn.Sequential(
        nn.Linear(2*hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions),
    )

    self.forward_model = nn.Sequential( 
        nn.Linear(hidden_size + 1, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size))


    for p in self.modules():
          if isinstance(p, nn.Linear):
              init.kaiming_uniform_(p.weight, a=1.0)
              p.bias.data.zero_()

  def input_preprocessing(self, state, next_state, action):
    state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
    next_state = torch.FloatTensor(np.array([s.__array__() for s in next_state])).to(self.device)
    action = torch.tensor(action, dtype=torch.int64).to(self.device)
    return state, next_state, action

  def forward(self, state, next_state, action):
    state, next_state, action = self.input_preprocessing(state, next_state, action)
    
    state = self.state_features_extractor(state)
    next_state = self.state_features_extractor(next_state)

    predicted_action = self.inverse_dynamics_model(torch.cat((state, next_state), 1)) 
    predicted_next_state = self.forward_model(torch.cat((state, action.unsqueeze(1)), 1))

    return next_state, predicted_next_state, action, predicted_action




class ActorCritic(nn.Module):
  def __init__(self, hidden_size, n_actions, device):
    super(ActorCritic, self).__init__()
    self.hidden_size = hidden_size
    self.n_actions = n_actions
    self.device = device
  
    self.state_features_extractor = FeatureExtractor(hidden_size).to(self.device)

    self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions))

    self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))
 
    for i in range(len(self.actor)):
      if type(self.actor[i]) == nn.Linear:
        init.orthogonal_(self.actor[i].weight, 0.01)
        self.actor[i].bias.data.zero_()

    
    for i in range(len(self.critic)):
      if type(self.critic[i]) == nn.Linear:
        init.orthogonal_(self.critic[i].weight, 0.01)
        self.critic[i].bias.data.zero_()

  
  def forward(self, state):
    state = torch.FloatTensor(np.array([s.__array__() for s in state])).to(self.device)
    state = self.state_features_extractor(state)
    policy = self.actor(state)
    value = self.critic(state)
    return policy, value
  