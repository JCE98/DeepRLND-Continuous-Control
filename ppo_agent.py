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

    def add(self, agent_index, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, prob_ratio=None, delta=None, advantage=None) # initizlize experience named tuple
        self.trajectories[agent_index].append(e)                                    # append experience named tuple to trajectory deque

    def flatten_experiences(self):
        for trajectory in self.trajectories:                                        # iterate over trajectories from each agent
            del trajectory[-1]                                                      # remove last element of trajectory segments with no advantage estimate
            self.memory.extend(trajectory)                                          # append trajectory to common experience deque

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        random.shuffle(self.memory)                                                 # shuffle common experience deque to avoid sampling from specific trajectories
        experiences = random.sample(self.memory, k=self.batch_size)                 # random sample from shuffled common experience deque
        tensor_kwargs = {'dtype':torch.float32, 'device':self.device, 'requires_grad':True} # keyword arguments for tensor generation

        states = torch.tensor(np.vstack([e.state for e in experiences if e is not None]), **tensor_kwargs)
        actions = torch.tensor(np.vstack([e.action for e in experiences if e is not None]), **tensor_kwargs)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences if e is not None]), **tensor_kwargs)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences if e is not None]), **tensor_kwargs)
        dones = torch.tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8), **tensor_kwargs)
        prob_ratios = torch.tensor(np.vstack([e.prob_ratio for e in experiences if e is not None]), **tensor_kwargs)
        advantages = torch.tensor(np.vstack([e.advantage for e in experiences if e is not None]), **tensor_kwargs)

        return (states, actions, rewards, next_states, dones, prob_ratios, advantages)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size, num_agents, args, random_seed):
        # agent/model construction parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # tensor casting device
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
        self.actor = Actor(state_size, action_size, random_seed, args.actorFCunits[0], args.actorFCunits[1])
        self.actor_old = Actor(state_size, action_size, random_seed, args.actorFCunits[0], args.actorFCunits[1])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic = Critic(state_size, action_size, random_seed, args.criticFCunits[0], args.criticFCunits[1])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(num_agents, action_size, args.trajectory_segment, args.minibatch_size, random_seed)

    def act(self, states):
        with torch.no_grad():
            [mu, sigma] = self.actor(torch.from_numpy(states))
        actions = deque(maxlen=self.num_agents*self.action_size)                # pre-allocate flattened actions container
        for loc, scale in zip(mu, sigma):                                       # iterate over actor output tensors for each agent
            for mean, std in zip(loc, scale):                                   # iterate over each pair in actor outputs
                dist = Normal(mean, std)                                        # form a Gaussian distribution with actor output
                actions.append(dist.sample().item())                            # add action from a sample of each Gaussian distribution to deque
        return np.array(actions).reshape(self.num_agents, self.action_size)     # reshape actions to pass back to environment
    
    def actionProbRatios(self, states, actions):
        """Determine probability ratios for current and old policies of selecting an action based on policy
        Params
        ======
            states (tensor): state from which the selected action was taken
            actions (tensor): action that was taken
        """
        probs = np.empty(actions.shape)                                         # initialize probabilities arrays
        probs_old = np.empty(actions.shape)
        for idx, policy in enumerate([self.actor, self.actor_old]):             # iterate over actor and actor_old policies
            prob = torch.empty(actions.shape)                                   # preallocate action probabilities tensor
            [mu, sigma] = policy(states)                                        # reconstruct Gaussian distribution parameters from states
            for index, (loc, scale) in enumerate(zip(mu, sigma)):               # iterate over pair in reconstruction
                dist = Normal(loc, scale)                                       # form a Gaussian distribution with reconstructed actor output
                prob[index] = dist.log_prob(actions[index])                     # fill in log probability of selecting action based on actor output reconstruction
            if idx == 0:
                probs = prob                                                    # assign probabilities to actor
            elif idx == 1:
                probs_old = prob                                                # assign probabilities to actor_old
            else:
                raise Exception('Probabilities not assigned!')
        return np.abs(probs - probs_old)                                        # the probability ratio is equivalent to the difference of the log probabilities

    def estimate_advantage(self):
        '''Estimate advantage of taking an action from a given state, as a measure of future reward from following the policy, based
            on trajectory information
        '''
        for idx, trajectory in enumerate(self.memory.trajectories):                                     # iterate over trajectories from all agents
            for jdx, experience in enumerate(trajectory):                                               # iterate over snapshots from each trajectory
                if jdx < self.trajectory_segment-1:                                                     # advantage can only be calculated out to T-1
                    state = torch.from_numpy(experience.state).to(self.device)
                    action = torch.from_numpy(experience.action).to(self.device)
                    next_state = torch.from_numpy(experience.next_state).to(self.device)
                    with torch.no_grad():
                        r = self.actionProbRatios(state, action)                                        # compute action probability ratios
                        next_action = torch.from_numpy(self.memory.trajectories[idx][jdx+1].action).to(self.device)
                        delta = r + self.gamma*self.critic(next_state, next_action) - self.critic(state, action)
                        self.memory.trajectories[idx][jdx] = experience._replace(prob_ratio=r, delta=delta) # fill in action probabilities and delta terms in experience named tuples by trajectory
        
        for idx, trajectory in enumerate(self.memory.trajectories):                                     # iterate over trajectories from all agents
            for jdx, experience in enumerate(trajectory):                                               # iterate over snapshots from each trajectory
                advantage = torch.zeros(delta.shape)                                                    # preallocate and initialize advantage estimate
                if jdx < self.trajectory_segment-1:                                                     # advantage can only be calculated out to T-1
                    delta = experience.delta
                    advantage += ((self.gamma*self.lambd)**(self.trajectory_segment-1-jdx))*delta       # calculate advantage
                    self.memory.trajectories[idx][jdx] = experience._replace(advantage=advantage)       # fill in advantage in snapshot trajectory tuples

    def step(self, states, actions, rewards, next_states, dones, optimize=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        if not optimize:
            # Save experience / reward
            for agent_index, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
                self.memory.add(agent_index, state, action, reward, next_state, done)
        
        if optimize:
            # Learn, if enough samples are available in memory
            self.estimate_advantage()                                           # calculate advantage estimates for snapshots in trajectory
            self.memory.flatten_experiences()                                   # flatten trajectories into common experience pool
            self.learn()

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
        Vpred = self.critic(states, actions)
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
        for eopch in range(self.optimization_epochs):
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones, prob_ratios, advantages = experiences
            for state, action, reward, prob_ratio, advantage in zip(states, actions, rewards, prob_ratios, advantages):
                Lclip = self.clipped_surrogate(prob_ratio, advantage)
                LVF_MSELoss = self.MSELoss(state, action, reward)
                loss = -torch.mean(Lclip - LVF_MSELoss)
                loss.backward()
                with torch.no_grad():
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
        self.actor_old.parameters = self.actor.parameters()        
    