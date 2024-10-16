import random, torch
import numpy as np
from torch.distributions import Normal
from collections import deque, namedtuple
from torch import optim
from model import Actor, Critic

class ReplayBuffer:
    def __init__(self, num_agents, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # tensor casting device
        self.action_size = action_size
        self.trajectories = [deque(maxlen=buffer_size) for idx in range(num_agents)] # internal memory (list of deques containing named tuple experiences)
        self.memory = deque(maxlen=num_agents*buffer_size)                           # deque of experience named tuples
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done", "prob_ratio", "delta", "advantage"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones, prob_ratios):
        """Add a new experience to memory."""
        for agent_index, (state, action, reward, next_state, done, prob_ratio) in enumerate(zip(states, actions, rewards, next_states, dones, prob_ratios)):
            e = self.experience(state, action, reward, next_state, done, prob_ratio, delta=None, advantage=None) # initizlize experience named tuple
            self.trajectories[agent_index].append(e)                                # append experience named tuple to trajectory deque

    def flatten_experiences(self):
        for trajectory in self.trajectories:                                        # iterate over trajectories from each agent
            del trajectory[-1]                                                      # remove last element of trajectory segments with no advantage estimate
            self.memory.extend(trajectory)                                          # append trajectory to common experience deque

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        random.shuffle(self.memory)                                                 # shuffle common experience deque to avoid sampling from specific trajectories
        experiences = random.sample(self.memory, k=self.batch_size)                 # random sample from shuffled common experience deque
        tensor_kwargs = {'dtype':torch.float32, 'device':self.device, 'requires_grad':True} # keyword arguments for tensor generation

        states = torch.vstack([e.state for e in experiences if e is not None])
        actions = torch.vstack([e.action for e in experiences if e is not None])
        rewards = torch.vstack([e.reward for e in experiences if e is not None])
        next_states = torch.vstack([e.next_state for e in experiences if e is not None])
        dones = torch.vstack([e.done for e in experiences if e is not None])
        prob_ratios = torch.vstack([e.prob_ratio for e in experiences if e is not None])
        advantages = torch.vstack([e.advantage for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones, prob_ratios, advantages)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size, num_agents, args, random_seed):
        # agent/model construction parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # tensor casting device
        self.tensor_kwargs = {'device':self.device, 'dtype':torch.float32, 'requires_grad':True}
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random_seed

        # learning parameters
        self.batch_size = args.minibatch_size
        self.optimization_epochs = args.optimization_epochs
        #self.buffer_size = args.buffer_size
        self.epsilon_clip = args.epsilon_clip
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.trajectory_segment = args.trajectory_segment

        # neural network models
        self.actor = Actor(state_size, action_size, random_seed, args.actorFCunits[0], args.actorFCunits[1]).to(self.device)
        self.actor_old = Actor(state_size, action_size, random_seed, args.actorFCunits[0], args.actorFCunits[1]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic = Critic(state_size, action_size, random_seed, args.criticFCunits[0], args.criticFCunits[1]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(num_agents, action_size, args.trajectory_segment, args.minibatch_size, random_seed)

    def act(self, states):
        self.actor.eval()
        self.actor_old.eval()
        with torch.no_grad():
            [mu, sigma] = self.actor(states)                                    # actor neural network output Gaussian probabilistic parameters
            [mu_old, sigma_old] = self.actor_old(states)
        self.actor.train()
        self.actor_old.eval()
        actions = torch.normal(mu, sigma)                                       # sample from Gaussian distribution to obtain actions
        dists = torch.distributions.Normal(mu,sigma)                            # construct normal distribution from Gaussian parameters
        dists_old = torch.distributions.Normal(mu_old,sigma_old)
        probs = dists.log_prob(actions)                                         # calculate log probabilities of selecting actions from Gaussian distributions
        probs_old = dists_old.log_prob(actions)
        prob_ratios = probs - probs_old                                         # probability ratio (equivalent to difference of log probs)
        
        return actions, prob_ratios
    
    def estimate_advantage(self):
        '''Estimate advantage of taking an action from a given state, as a measure of future reward from following the policy, based
            on trajectory information
        '''
        for idx, trajectory in enumerate(self.memory.trajectories):                                     # iterate over trajectories from all agents
            for jdx, experience in enumerate(trajectory):                                               # iterate over snapshots from each trajectory
                if jdx < self.trajectory_segment-1:                                                     # advantage can only be calculated out to T-1
                    state = experience.state
                    action = experience.action
                    next_state = experience.next_state
                    r = experience.prob_ratio                                                           # compute action probability ratios
                    next_action = self.memory.trajectories[idx][jdx+1].action
                    self.critic.eval()
                    with torch.no_grad():
                        V_state = self.critic(state, action)
                        V_next_state = self.critic(next_state, next_action)
                    self.critic.train()
                    delta = r + self.gamma*V_next_state - V_state
                    self.memory.trajectories[idx][jdx] = experience._replace(delta=delta)               # fill in action probabilities and delta terms in experience named tuples by trajectory
        
        for idx, trajectory in enumerate(self.memory.trajectories):                                     # iterate over trajectories from all agents
            for jdx, experience in enumerate(trajectory):                                               # iterate over snapshots from each trajectory
                advantage = torch.zeros(delta.shape, **self.tensor_kwargs)                              # preallocate and initialize advantage estimate
                if jdx < self.trajectory_segment-1:                                                     # advantage can only be calculated out to T-1
                    delta = experience.delta
                    advantage = advantage + ((self.gamma*self.lambd)**(self.trajectory_segment-1-jdx))*delta # calculate advantage
                    self.memory.trajectories[idx][jdx] = experience._replace(advantage=advantage)       # fill in advantage in snapshot trajectory tuples

    def clipped_surrogate(self, prob_ratio, advantage):
        '''Calculate the clipped surrogate function for a given action
        Params
        ======
            probs       (tensor): probability ratios that actions would be taken under the new policy, as opposed to the previous one
            advantages  (tensor): advantage estimate of having taken that action, as a measure of future reward from following this policy
        '''
        Lclip = torch.min(torch.mul(prob_ratio,advantage),torch.mul(torch.clip(prob_ratio, 1-self.epsilon_clip, 1+self.epsilon_clip),advantage))
        return Lclip
    
    def MSELoss(self, states, actions, rewards):
        reward = torch.mean(rewards)
        self.critic.eval()
        with torch.no_grad():
            Vpred = self.critic(states, actions)
        self.critic.train()
        mseloss = (reward - Vpred)**2
        return mseloss

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor(state) -> action
            critic(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, prob_ratio, delta, advantage) tuples 
            gamma (float): discount factor
        """
        self.estimate_advantage()                                           # calculate advantage estimates for snapshots in trajectory
        self.memory.flatten_experiences()                                   # flatten trajectories into common experience pool
        for epoch in range(self.optimization_epochs):
            self.actor_optimizer.zero_grad()                                # set optimizer gradients back to zero
            self.critic_optimizer.zero_grad()
            loss = torch.zeros(1, **self.tensor_kwargs)                     # initialize loss tensor
            experiences = self.memory.sample()                              # random sample experiences with batch_size
            states, actions, rewards, next_states, dones, prob_ratios, advantages = experiences
            for state, action, reward, prob_ratio, advantage in zip(states, actions, rewards, prob_ratios, advantages):
                Lclip = self.clipped_surrogate(prob_ratio, advantage)       # clipped surrogate function
                LVF_MSELoss = self.MSELoss(state, action, reward)           # state-action value function MSE loss
                loss = loss + (Lclip - LVF_MSELoss)
            loss = -torch.mean(loss/self.batch_size)                        # empirical average over a finite batch of samples
            loss.backward(retain_graph=True)                                # backpropogation through computational graph
            with torch.no_grad():
                self.actor_optimizer.step()                                 # calculate gradients of neural net parameters and take step
                self.critic_optimizer.step()
        actor_weights = self.actor.state_dict()
        actor_old_weights = self.actor_old.state_dict()
        for name, param in actor_weights.items():
            actor_old_weights[name].copy_(param)                            # actor_old -> actor
        self.actor_old.load_state_dict(actor_old_weights)        
    