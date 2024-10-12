# import libraries
import random, torch, math, time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from torch import optim
from torch import nn
from policy import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trajectories(TensorDataset):
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
        self.tensor_kwargs = {'device':device, 'dtype':torch.float32, 'requires_grad':True}

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
        [mu, sigma] = self.actor.forward(states)                                # pass states to actor network to obtain Gaussian parameters
        actions = torch.tensor([], **self.tensor_kwargs)                        # pre-allocate actions tensor
        for loc, scale in zip(mu, sigma):
            dist = Normal(loc, scale)                                           # form a Gaussian distribution with actor output
            actions = torch.cat((actions,dist.sample()))                        # sample from the distribution to obtain action
        return torch.reshape(actions, (self.num_agents,self.action_size)).to('cpu')

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
            sars = (states[i,:],
                    actions[i,:],
                    rewards[i],
                    next_states[i,:])                                       # preprocess tensors from environment to tuples of tensors
            self.trajectories[i].append(sars)                               # trajectories list consists of trajectory lists of snapshot tuples

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r\t\t{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

    def actionProbRatios(self,states,actions):
        """Determine probability of selecting an action based on policy
        Params
        ======
            states (tensor): state from which the selected action was taken
            actions (tensor): action that was taken
        """
        probs = torch.empty(actions.shape, **self.tensor_kwargs)              # initialize probabilities arrays
        probs_old = torch.empty(actions.shape, **self.tensor_kwargs)
        for idx, policy in enumerate([self.actor, self.actor_old]):
            prob = torch.tensor([], **self.tensor_kwargs)
            [mu, sigma] = policy.forward(states)
            for index, (loc, scale) in enumerate(zip(mu, sigma)):
                dist = Normal(loc, scale)
                prob = torch.cat((prob, torch.reshape(dist.log_prob(actions[index]), (1,))))
            if idx == 0:
                probs = prob
            elif idx == 1:
                probs_old = prob
            else:
                raise Exception('Probabilities not assigned!')
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
        print('\tOptimizing policy...')
        start = time.time()
        self.estimate_advantage()                                   # advantage estimate for each snapshot
        print(f'\t\tAdvantage estimate took {time.time()-start} seconds')
        start = time.time()
        self.experience = Trajectories(self.trajectories)           # build dataset for minibatch sampling
        dataloader = DataLoader(dataset=self.experience, 
                                batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)        # instantiate dataloader to perform minibatch sampling
        print(f'\t\tLoading experiences into replay buffer took {time.time()-start} seconds')
        # Optimizers
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        start = time.time()
        self.printProgressBar(0, self.optimization_epochs, prefix='Progress:',suffix = 'Complete', length = 50)
        for epoch in range(self.optimization_epochs):
            Lclip = torch.tensor([], **self.tensor_kwargs)          # preallocating loss function constituent tensors
            LVF = torch.tensor([], **self.tensor_kwargs)
            for i, (states, probs, advantages) in enumerate(dataloader):
                states = states.to(device)
                probs = probs.to(device)
                advantages = advantages.to(device)
                Lclip = torch.cat((Lclip, self.clipped_surrogate(probs, advantages)))
                LVF = torch.cat((LVF, self.MSELoss(states, advantages)))# MSE loss for state/action value estimate
            loss = -torch.mean(Lclip - LVF)                         # combining clipped surrogate function and MSE value function loss
            loss.backward()                                         # backward pass
            with torch.no_grad():
                actor_optimizer.step()                              # weights update
                critic_optimizer.step()
            actor_optimizer.zero_grad()                             # empty gradients for next iteration
            critic_optimizer.zero_grad()
            self.printProgressBar(epoch+1, self.optimization_epochs, prefix='Progress:', suffix = 'Complete', length = 50)
        self.actor_old.parameters = self.actor.parameters()
        print(f'\t\tOptimization over {self.optimization_epochs} epochs, with batch size {self.batch_size} took {time.time()-start} seconds')
        self.clear_buffer()                                         # empty trajectories for next segment