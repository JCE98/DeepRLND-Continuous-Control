# import libraries
import random, torch, math
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from torch import optim
from torch import nn
from policy import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trajectories(Dataset):
    def __init__(self,trajectories):
        '''
        Params
        ======
        Name            |       Type                |       Definition                  |       Shape
        =====================================================================================================================
        self               Trajectories(Dataset)        parent object                       N/A
        trajectories       list(list(tuples))           agent experiences*                  num_agents,trajectory_segment,7
        
        *Agent experiences consist of a list of trajectories. These trajectories are encapsulated as a list of tuples, containing
        SARS information, along with action probabilities, delta terms, and advantage estimates
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.prob_ratios = []
        self.deltas = []
        self.advantages = []
        for trajectory in trajectories:
            for timestep in trajectory:
                self.states.append(timestep[0].detach())
                self.actions.append(timestep[1].detach())
                self.rewards.append(timestep[2].detach())
                self.next_states.append(timestep[3].detach())
                self.prob_ratios.append(timestep[4].detach())
                self.deltas.append(timestep[5].detach())
                self.advantages.append(timestep[6].detach())
        self.n_samples = len(self.states)

    def __getitem__(self, index):
        return (self.states[index], self.prob_ratios[index], self.advantages[index])

    def __len__(self):
        return self.n_samples


class Agent():
    def __init__(self,state_size,action_size,num_agents,args,seed):
        """Initialize an Agent object.
        
        Params
        ======
            Name                |   Type    |   Definition                  
            ======================================================================================
            self                    Agent       agent object being instantiated
            state_size              int         number of states tracked in environment
            action_size             int         number of actions for each agent
            num_agents              int         number of parallel agents
            args                    struct      information for design of model and optimization
            seed                    int         random seed for agent initialization
            
        Output
        ======
            Instantiated Agent object
        """
        # general
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.trajectory_segment = args.trajectory_segment
        self.seed = random.seed(seed)
        
        # neural networks
        self.actor = Actor(state_size, action_size, seed, fc1_units=args.actorFCunits[0], fc2_units=args.actorFCunits[1]).to(device)
        self.actor_old = Actor(state_size, action_size, seed, fc1_units=args.actorFCunits[0], fc2_units=args.actorFCunits[1]).to(device)
        self.critic = Critic(state_size, action_size, seed, fc1_units=args.criticFCunits[0], fc2_units=args.criticFCunits[1]).to(device)

        # experience collection
        self.trajectories = [[] for _ in range(num_agents)]

        # optimization parameters
        self.batch_size = args.minibatch_size
        self.learning_rate = args.learning_rate
        self.optimization_epochs = args.optimization_epochs
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.epsilon = args.epsilon_clip


    def act(self,states):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            self (Agent): parent object
            states (np.ndarray): current state to determine agent action
        Output
        ======
            actions (tensor[num_agents,action_size]): actions probabilistically selected based on current state
        """
        state = torch.tensor(states)                                            # convert state array to tensor
        [mu, sigma] = self.actor.forward(state)                                 # pass states to actor network to obtain Gaussian parameters
        actions = torch.empty(mu.shape)                                         # pre-allocate actions tensor
        for index, (loc, scale) in enumerate(zip(mu, sigma)):
            dist = Normal(loc, scale)                                           # form a Gaussian distribution with actor output
            actions[index] = dist.sample()                                      # sample from the distribution to obtain action
        #actions = self.actor.act(states)                                       # get actions from policy
        return actions

    def build_trajectory(self, states, actions, rewards, next_states):
        '''Builds trajectory experience from stored numpy arrays of agent experiences
        Params
        ======
            self (Agent): parent object
            states (tensor[num_agents,state_size]): array of states per agent
            actions (tensor[num_agents,action_size]): array of actions per agent
            rewards (tensor[num_agents,]): array of rewards per agent
            next_states (tensor[num_agents,state_size]): array of next states per agent

        List of trajectories consists of lists of tuples, containing SARS information at each timestep
        '''
        for i in range(len(self.trajectories)):
            sars = (torch.tensor(states[i,:], requires_grad=True),
                    torch.tensor(actions[i,:], requires_grad=True),
                    torch.tensor(rewards[i], requires_grad=True),
                    torch.tensor(next_states[i,:], requires_grad=True))     # convert numpy arrays from environment to tuples of tensors
            self.trajectories[i].append(sars)                               # trajectories list consists of a lists of tuples

    def actionProbRatios(self,states,actions):
        """Determine probability of selecting an action based on policy
        Params
        ======
            states (tensor): state from which the selected action was taken
            actions (tensor): action that was taken
        """
        probs = torch.empty(actions.shape)                                  # pre-allocate probabilities arrays
        probs_old = torch.empty(actions.shape)
        for policy, prob in zip([self.actor, self.actor_old], [probs, probs_old]):
            [mu, sigma] = policy.forward(states)
            for index, (loc, scale) in enumerate(zip(mu, sigma)):
                dist = Normal(loc, scale)
                prob[index] = dist.log_prob(actions[index])
        return probs - probs_old
    
    def estimate_advantage(self):
        '''Estimate advantage of taking an action from a given state, as a measure of future reward from following the policy, based
            on trajectory information
        '''
        for trajnum, trajectory in enumerate(self.trajectories):
            for timestep, sars in enumerate(trajectory):
                states = sars[0]
                actions = sars[1]
                next_states = sars[3]
                r = self.actionProbRatios(states,actions)
                delta = r + self.gamma*self.critic.forward(next_states) - self.critic.forward(states)
                self.trajectories[trajnum][timestep] += (r,delta,)            # add action probability ratio and delta terms to sars tuple
        for trajnum, trajectory in enumerate(self.trajectories):
            for timestep in range(self.trajectory_segment-1):
                advantage = 0
                for t in range(timestep,self.trajectory_segment-1):
                    delta = trajectory[t][5]
                    advantage += ((self.gamma*self.lambd)**(self.trajectory_segment-1-t))*delta
                self.trajectories[trajnum][timestep] += (advantage,)          # add advantage to snapshot trajectory tuples

    def clipped_surrogate(self, probs, advantages):
        '''Calculate the clipped surrogate function for a given action
        Params
        ======
            probs       (tensor): probability ratios that actions would be taken under the new policy, as opposed to the previous one
            advantages  (tensor): advantage estimate of having taken that action, as a measure of future reward from following this policy
        '''
        Lclip = torch.min(torch.mul(probs,advantages),torch.mul(torch.clip(probs, 1-self.epsilon, 1+self.epsilon),advantages))
        return Lclip
    
    def MSELoss(self, states, advantages):
        '''Compute the MSE Loss for the learned value function critic network
        Params
        =======
            states      (tensor): state to evaluate the value function network
            advantages  (tensor): advantage estimates for that state based on the current policy
        '''
        with torch.no_grad():
            values = self.critic.forward(states)                    # obtain state/action value estimate from learned value function network
        mse = (values - advantages)**2                              # calculate MSE loss between the estimates
        return mse
    
    def clear_buffer(self):
        '''Empty trajectories to prepare for the next set of experiences'''
        self.trajectories = [[] for _ in range(self.num_agents)]

    def learn(self):
        '''Perform minibatch sampling based optimization of actor and value function networks'''
        self.estimate_advantage()                                   # advantage estimate for each snapshot
        self.experience = Trajectories(self.trajectories)           # build dataset for minibatch sampling
        dataloader = DataLoader(dataset=self.experience, 
                                batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)        # instantiate dataloader to perform minibatch sampling
        # Optimizers
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        for epoch in range(self.optimization_epochs):
            Lclip = torch.tensor([], requires_grad=True)            # preallocating loss function constituent tensors
            LVF = torch.tensor([], requires_grad=True)
            with tqdm(total=self.batch_size, position=0, leave=True) as pbar:
                for i, (states, probs, advantages) in tqdm(enumerate(dataloader), position=0, leave=True):
                    Lclip = torch.cat((Lclip, self.clipped_surrogate(probs, advantages)))
                    LVF = torch.cat((LVF, self.MSELoss(states, advantages)))# MSE loss for state/action value estimate
                    pbar.update()
            loss = torch.mean(Lclip - LVF)                          # combining clipped surrogate function and MSE value function loss
            loss.backward()                                         # backward pass
            with torch.no_grad():
                actor_optimizer.step()                              # weights update
                critic_optimizer.step()
            actor_optimizer.zero_grad()                             # empty gradients for next iteration
            critic_optimizer.zero_grad()
            #pbar.set_description(f"epoch {epoch+1}/{self.optimization_epochs}", refresh=True)
            #pbar.update()
        self.clear_buffer()                                         # empty trajectories for next segment