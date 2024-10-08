#Import Libraries
import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)                             # input layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)                              # fully connected layer
        self.fc3 = nn.Linear(fc2_units, action_size)                            # output layer

    def forward(self, state):
        """Build a network that maps state -> action selection.
        Params
        ======
            state (tensor): state input to actor neural network
        """
        state = state.float()
        x = F.relu(self.fc1(state))                                             # input layer with relu activation 
        x = F.relu(self.fc2(x))                                                 # fully connected layer with relu activation
        mu = F.tanh(self.fc3(x))                                                # output layer with tanh activation (Gaussian location)
        sigma = F.sigmoid(self.fc3(x))                                          # output layer with sigmoid activation (Gaussian scale)
        return mu, sigma
    '''
    def act(self, state):
        """Select action based on policy
        Params
        ======
            state (tensor): state input to actor neural network
        """
        [mu, sigma] = self.forward(state)                                       # pass states to actor network to obtain Gaussian parameters
        actions = torch.empty(mu.shape)                                            # pre-allocate actions array
        for index, (loc, scale) in enumerate(zip(mu, sigma)):
            dist = Normal(loc, scale)
            actions[index] = dist.sample()
        return actions
    
    def probs(self, state, actions):
        """Determine probability of selecting an action based on policy
        Params
        ======
            state (tensor): state from which the selected action was taken
            action (array): action that was taken
        """
        probs = torch.empty(actions.shape)                                # pre-allocate probabilities array
        [mu, sigma] = self.forward(state)                                 # pass states to actor network to obtain Gaussian parameters
        for index, (loc, scale) in enumerate(zip(mu, sigma)):     
            dist = Normal(loc, scale)                                     # create Gaussian distributions
            probs[index] = dist.log_prob(actions[index])                  # sample from distributions to obtain actions
        return probs
    '''
class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build neural network model for action value function approximation
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)                     # input layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)                      # fully connected layer
        self.fc3 = nn.Linear(fc2_units, action_size)                    # action value estimate output

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = state.float()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)