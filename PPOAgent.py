# import libraries
import random, torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import optim
from policy import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self,state_size,action_size,num_agents,trajectory_segment,learning_rate,actorFCunits,criticFCunits,seed):
        """Initialize an Agent object.
        
        Params
        ======
            Name                |   Type    |   Definition                  
            ====================================================================
            self                    Agent       agent object being instantiated
            state_size              int         number of states tracked in environment
            action_size             int         number of actions for each agent
            num_agents              int         number of parallel agents
            trajectory_segment      int         Length of fixed trajectory segment
            learning_rate           float       learning rate for agent optimization
            actorFCunits            list        number of neurons in actor nn FC layers
            criticFCunits           list        number of neurons in critic nn FC layers
            seed                    int         random seed for agent initialization
            
        Output
        ======
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.trajectory_segment = trajectory_segment
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate

        # neural networks
        self.actor = Actor(state_size, action_size, seed, fc1_units=actorFCunits[0], fc2_units=actorFCunits[1]).to(device)
        self.actor_old = Actor(state_size, action_size, seed, fc1_units=actorFCunits[0], fc2_units=actorFCunits[1]).to(device)
        self.critic = Critic(state_size, action_size, seed, fc1_units=criticFCunits[0], fc2_units=criticFCunits[1]).to(device)

        # replay buffer
        self.states = np.zeros((num_agents,self.state_size,trajectory_segment))
        self.actions = np.zeros((num_agents,self.action_size,trajectory_segment))
        self.rewards = np.zeros((num_agents,1,trajectory_segment))

    def act(self,state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            Name    |   Type    |   Definition                                          |   Shape 
            ===========================================================================================
            self        Agent       parent object on which this method is being called      N/A
            state       array       current state to determine agent action                 state_size
            
        Output
        ======
            Name            |   Type    |   Definition                                                  |   Shape
            ===========================================================================================================
            actionsarray        array       actions probabilistically selected based on current state       action_size
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)                 # convert state vector to torch format
        self.actor.eval()                                                               # put actor in evaluation mode
        with torch.no_grad():                                                           # turn off gradient computation
            actions = self.actor.act(state)                                             # get action values from policy
        self.actor.train()                                                              # put actor in training mode
        return actions

    def build_trajectory(self, states, actions, rewards, idx):
        self.states[:,:,idx] = states
        self.actions[:,:,idx] = actions
        self.rewards[:,:,idx] = rewards

    def step(self,args):
        pass
        #TODO: implement trajectory segment optimization
        

    def clearBuffer(self):
        self.states = np.zeros((self.num_agents,self.state_size,self.trajectory_segment))
        self.actions = np.zeros((self.num_agents,self.action_size,self.trajectory_segment))
        self.rewards = np.zeros((self.num_agents,1,self.trajectory_segment))