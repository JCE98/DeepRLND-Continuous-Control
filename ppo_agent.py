#Import Libraries
import random, torch
from scipy.stats import norm
import numpy as np
from model import Actor, Critic
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Agent Class
class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, learning_rate, actorFCunits, criticFCunits, seed):
        """Initialize an Agent object.
        
        Params
        ======
            Name        |   Type    |   Definition                  
            ====================================================================
            self            Agent       agent object being instantiated
            state_size      int         number of states tracked in environment
            action_size     int         number of actions for each agent
            learning_rate   float       learning rate for agent optimization
            actorFCunits    list        number of neurons in actor nn FC layers
            criticFCunits   list        number of neurons in critic nn FC layers
            seed            int         random seed for agent initialization
            
        Output
        ======
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Neural Networks
        self.actor = Actor(state_size, action_size, seed, fc1_units=actorFCunits[0], fc2_units=actorFCunits[1]).to(device)
        self.actor_old = Actor(state_size, action_size, seed, fc1_units=actorFCunits[0], fc2_units=actorFCunits[1]).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = Critic(state_size, action_size, seed, fc1_units=criticFCunits[0], fc2_units=criticFCunits[1]).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def act(self, state):
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
            action_values = self.actor(state)                                           # get action values from policy
        self.actor.train()                                                              # put actor in training mode
        action_values = action_values.numpy()
        # Probabilistic Action Selection
        actionsarray = np.array([]).reshape([0,action_values[0].shape[1]])
        for output in action_values[0]:
            actions = np.array([])
            for index in range(self.action_size):
                actions = np.append(actions,np.random.normal(output[2*index],output[2*index+1]))   # determine actions based on policy action values of mean and standard deviation
            actionsarray = np.vstack(actions)
        return actionsarray
    
    def policy_ratios(self, states, actions):
        """Returns ratios of probabilities for the new and old policies to return the actions taken for a given state
        
        Params
        ======
            Name    |   Type    |   Definition                                          |   Shape (Dim1)    |   Shape (Dim2)
            ===========================================================================================================================
            self        Agent       parent object on which this method is being called      N/A                 N/A
            states      array       states experienced by agent at each iteration           state_size          args.trajectory_segment
            actions     array       actions taken by the agent at each iteration            action_size         args.trajectory_segment

        Output
        ======
            Name            |   Type    |   Definition                                                                                      |   Shape (Dim1)    |   Shape (Dim2)
            ===============================================================================================================================================================================
            policy_ratios       array       ratios of the probability of action selection from each state between current and old policies      action_size         args.trajectory_segment

        Notes
        =====
            policy_prime (Agent.actor or Agent.critic): neural network new weights
            policy_old (Agent.actor or Agent.critic): neural network old weights
        """
        policy_old = self.actor_old
        policy_prime = self.actor
        probs = np.array([]).reshape(0,self.action_size)                                # initialize trajectory action proabilities array
        for index, policy in enumerate([policy_prime, policy_old]):
            probs = np.array([]).reshape((0,self.action_size))
            for state, action in zip(states, actions):
                torchstate = torch.from_numpy(state).float().unsqueeze(0).to(device)
                policy.eval()                                                           # put actor in evaluation mode
                with torch.no_grad():                                                   # turn off gradient computation
                    action_values = policy(torchstate)                                  # get action values from policy
                policy.train()                                                          # put actor in training mode
                action_values = action_values.numpy()
                means = action_values[0,0::2]                                           # extract probabilistic action distribution means
                sds = action_values[0,1::2]                                             # extract probabilistic action distribution standard deviations
                prob = np.array([norm.pdf(act, loc=mean, scale=sd) for mean,sd,act in zip(means, sds, action)]) # probability from a normal distribution that the action would be selected
                probs = np.vstack([probs,prob.reshape((1,self.action_size))])
            if index==0: 
                probs_prime = probs
            else:
                probs_old = probs
        return np.divide(probs_prime,probs_old).transpose()
    
    def advantage(self, states, actions, params):
        """Returns the advantage estimate for each timestep in a trajectory generated by an agent/environment interaction episode.
        
        Params
        ======
            Name    |   Type    |   Definition                                  |   Shape (Dim1)    |   Shape (Dim2)            |   Shape (Dim3)
            ========================================================================================================================================
            states      array     history of states for trajectory segment          state_size          args.trajectory_segment     N/A
            actions     array     history of actions for trajectory segment         action_size         args.trajectory_segment     N/A
            params      dict      commmand line options and default values          N/A                 N/A                         N/A

        Output
        ======
            Name    |   Type    |   Definition                                  |   Shape (Dim1)    |   Shape (Dim2)            |   Shape (Dim3)
            ========================================================================================================================================
            advantage   array     advantage estimates for trajectory segment        action_size         args.trajectory_segment     N/A
        """
        deltas = np.empty((self.action_size,))
        valueFcn = np.array([])
        rt = self.policy_ratios(np.transpose(states), np.transpose(actions))            # calculate the iterative policy probability ratio for choosing these actions from these states
        for index, state in enumerate(np.transpose(states)):
            if index < states.shape[1]:                                                 # exclude last index, since there is no next state
                torchstate = torch.from_numpy(state).float().unsqueeze(0).to(device)    # convert from numpy array to torch tensor
                #torchnextstate = torch.from_numpy(states[:,index+1]).float().unsqueeze(0).to(device)
                self.critic.eval()                                                      # put critic in eval mode
                with torch.no_grad():
                    critic_output = self.critic(torchstate)                             # evaluate critic at current state
                    #critic_output_next_state = self.critic(torchnextstate)              # evaluate critic at next state
                self.critic.train()                                                     # return critic to training mode
                #critic_output = critic_output.item()                                    # convert torch tensor output to float
                #critic_output_next_state = critic_output_next_state.item()
                valueFcn = np.append(valueFcn,critic_output.item())
        for index in range(states.shape[1]-1):
            deltas = np.vstack((deltas, rt[:,index] + params.gamma*valueFcn[index+1] - valueFcn[index]))
        deltas = deltas.transpose()
        discount = np.array([(params.lambd*params.gamma)**exp for exp in range(states.shape[1])])
        advantages = np.multiply(deltas,discount)

                #deltas = np.vstack((deltas,rt[:,index].transpose() + params.gamma*critic_output_next_state - critic_output)) # delta terms at each iteration
        #advantages = np.multiply(deltas.transpose(),np.array([(params.lambd*params.gamma)**n for n in range(deltas.shape[0])]).transpose()).sum() # advantage estimates at each iteration
        return advantages

    def clipped_surrogate(self, states, actions, advantages, epsilon=0.1):
        """Returns the clipped surrogate function comparing a new policy to an old one
        
        Params
        ======
            states (array-like): history of the environment states from which the agent selected it's actions
            actions (array-like): history of the actions selected by the agent
            advantages (array-like): advantage estimates at each time step
            epsilon (float): clipping limit for surrogate function
        """
        Lclip = np.zeros(1,self.action_size)
        ratios = self.policy_ratios(self.actor, self.actor_old, states, actions)        # calculate probability ratios of current and old policies to take the actions at the given states
        for ratio, advantage in zip(ratios, advantages):
            Lclip.dstack((Lclip,[min(ratio[index]*advantage[index], max(1-epsilon,min(ratio[index],1+epsilon))*advantage[index]) for index in len(ratio)]))
        return Lclip[:,:,1:].reshape(1,self.action_size,len(ratios))

    def expectation_eval(self, states, rewards):
        pass
        #TODO: Implement value function vs. future reward comparison loss function


    def optimize(self, states, actions, rewards, advantages, epsilon, minibatch_size, optimization_epochs):
        """Update actor and critic weights using the clipped surrogate function

        Params
        ======
           states (array-like): history of the environment states from which the agent selected it's actions
           actions (array-like): history of the actions selected by the agent
           advantages (array-like): advantage estimates at each time step
           epsilon (float): surrogate function clipping limit 
        """
        for epoch in range(optimization_epochs):
            #TODO: Implement minibatch sampling for actor/critic optimization
            #TODO: Implement policy and value network optimization
            self.actor.compile(optimizer=self.actor_optimizer, loss=-self.clipped_surrogate(states, actions, advantages, epsilon))
            self.critic.compile(optimizer=self.critic_optimizer, loss=-self.expectation_eval(states, rewards))
        return 0
        