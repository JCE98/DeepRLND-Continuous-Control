'''
Udacity Deep Reinforcement Learning Nanodegree
Continuous Control Project
Author: Josiah Everhart
Objective: Train a deep reinforcement learning agent to keep the end effector of a robotic arm within a moving target area. 
Implement PPO algorithm as described in ref [1]
'''

#Import Libraries
import torch, os, argparse
import numpy as np
from unityagents import UnityEnvironment
from ppo_agent import Agent

def checkArguments(args):
    if ((args.trajectory_segment is not None and args.episode_max is not None) and (args.trajectory_segment > args.episode_max)):
        argparse.ArgumentTypeError('Trajectory segment length provided is longer than the episode maximum number of iterations')
    if args.mode != 'training' and args.mode != 'demo':
        argparse.ArgumentTypeError('Invalid mode specified. Available modes are training and demo.')
    if args.lambd < 0 or args.lambd > 1:
        argparse.ArgumentTypeError('Lambda value must be between 0 and 1')
    if args.gamma < 0 or args.gamma > 1:
        argparse.ArgumentTypeError('Gamma value must be between 0 and 1')

def argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for PPO agent training on Reacher application")
    parser.add_argument("mode", type=str, nargs='?',default="training", help="training or demo mode")
    parser.add_argument("gamma", type=float, nargs='?', default=0.95, help="discount factor for future rewards, between 0 and 1")
    parser.add_argument("lambd", type=float, nargs='?', default=0.9, help="advantage estimate discount factor, between 0 and 1")
    parser.add_argument("training_episodes", type=int, nargs='?', default=2000, help="number of episodes to train agents")
    parser.add_argument("episode_max", type=int, nargs='?', default=1000, help="maximum number of iterations to run the episode")
    parser.add_argument("learning_rate", type=float, nargs='?', default=1E-5, help="learning rate for agent optimization")
    parser.add_argument("trajectory_segment", type=int, nargs='?', default=100)
    parser.add_argument("epsilon_clip", type=float, nargs='?', default=0.1, help="clipping value for optimization surrogate function")
    parser.add_argument("optimization_epochs", type=int, nargs='?', default=10, help="number of epochs over which to optimize the surrogate function")
    parser.add_argument("minibatch_size", type=int, nargs='?', default=500, help="minibatch size for surrogate function optimization")
    parser.add_argument("actorFCunits", type=list, nargs='?', default=[64,64], help="number of neurons in each FC layer of the actor neural network")
    parser.add_argument("criticFCunits", type=list, nargs='?', default=[64,64], help="number of neurons in each FC layer of the critic neural network")
    args = parser.parse_args()
    #Input processing / error handling
    checkArguments(args)
    return args

def ppo(env, args):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]                                                          # environment reset
    states = env_info.vector_observations                                                                       # get the current state (for each agent)
    state_size = states.shape[1]                                                                                # state space size of environment
    action_size = brain.vector_action_space_size                                                                # action space size of environment
    num_agents = len(env_info.agents)                                                                           # number of agents to be trained
    scores = np.zeros(num_agents)                                                                               # initialize the score (for each agent)
    log = {}                                                                                                    # dictionary for logging
    #agents = [Agent(state_size, action_size, args.learning_rate, args.actorFCunits, args.criticFCunits, seed=0) \
              #for i in range(num_agents)]                                                                       # instantiate each agent
    agent = Agent(state_size, action_size, args.learning_rate, args.actorFCunits, args.criticFCunits, seed=0)
    log['scores'] = np.array([])
    for i in range(1,args.training_episodes+1):                                                                 # episode loop
        scores = np.zeros(num_agents)                                                                           # initialize episode score
        env_info = env.reset(train_mode=True)[brain_name]                                                       # environment reset in training mode
        states = env_info.vector_observations                                                                   # get the current states
        actions = np.zeros([num_agents, action_size])                                                             # initialize action array
        iters = 1                                                                                               # initialize iteration counter
        while iters<=args.episode_max:                                                                          # iterations loop
            log['states'] = np.array(states)                                                                    # reset log to only include current trajectory segment
            log['actions'] = np.array([]).reshape([0,action_size])
            log['rewards'] = np.array([])
            log["advantage"] = np.array([])
            while iters%args.trajectory_segment != 0:                                            # fixed length trajectory segments
                actions = agent.act(states)                                                        # get actions from agent, according to current policy
                log["actions"] = np.dstack((log["actions"], actions))                                            # log actions for each agent at each timestep
                env_info = env.step(actions)[brain_name]                                                        # change the environment as a result of the actions and time step
                next_states = env_info.vector_observations                                                      # new states after environment step
                dones = env_info.local_done                                                                     # flags for episode completion
                scores +=env_info.rewards                                                                       # increment episode score using environment reward
                log["rewards"] = np.stack((log["rewards"],env_info.rewards))                                    # collect rewards at each time step
                log["states"] = np.stack((log["states"],states))                                                # log states at each timestep
                states = next_states                                                                            # set the state for the next timestep
                iters +=1                                                                                       # increment the iteration counter
                if any(dones):
                    break                                                                                       # exit trajectory segment loop
            if any(dones):
                break                                                                                           # exit episode loop
            for index, agent in enumerate(agents):                                                              # parallel agents
                log["advantage"][:,index,:] = agent.advantage(log["states"][:,index,:], log["actions"][:,index,:], \
                                                     lambd=args.lambd, gamma=args.gamma)                        # advantage estimate over the trajectory segment for each agent
            for index, agent in enumerate(agents):
                agent.optimize(log["states"], log["actions"], log["rewards"], log["advantages"], \
                               args.epsilon_clip, args.minibatch_size, args.optimization_epochs)                # optimize the actor wrt the NN weights, relative to the clipped surrogate function
                agent.actor_old = agent.actor                                                                   # set the old policy to the current policy for the next trajectory segment
        log["scores"] = np.stack((log["scores"],scores))                                                        # log scores of each agent per episode
    return log["scores"]

#TODO: Implement trained agent demonstration
def demoTrainedAgent(env, agentFile, n_episodes=100, max_t=200):
    #Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    #Reset the environment in training mode
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations                                                                       # get the current state (for each agent)
    state_size = states.shape[1]
    scores = np.zeros(num_agents)                                                                               # initialize the score (for each agent)
    action_size = brain.vector_action_space_size
    agent = torch.load(agentFile)
    for i in range(1,n_episodes+1):
        for t in range(max_t):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations                                                               # get the current state
            actions = agent.act(states)                                                                         # get action from agent, according to trained policy
            env_info = env.step(actions)[brain_name]                                                            # change the environment as a result of the action and the passage of time



if __name__=="__main__":
    args = argparser()                                                                                          # parse command line arguments
    #Instantiate Unity Environment
    unityPackagePath = os.path.join(os.path.dirname(__file__),'Reacher_Windows_x86_64/Reacher.exe')             # path to Unity environment executable
    env = UnityEnvironment(file_name = unityPackagePath)                                                        # instantiate Unity environment
    #Training or Demonstration
    if args.mode == "training":
        scores = ppo(env, args)
    elif args.mode == "demo":
        while True:
            agentFile = input("Enter the path to the trained agent checkpoint .pth file: ").replace("\\","/")
            if not os.path.exists(agentFile):                                                               # if the file does not exist
                print("The file path you entered could not be found. Please check the path and try again.\n")
            else:
                print("Loading agent checkpoint at %s..." % agentFile)
                break
        scores = demoTrainedAgent(env,agentFile)
    else:
        raise Exception("Invalid mode selected. Please choose training or demo mode.")
    filename = args.mode
    env.close()



#References
"""
[1] Schulman, John; Wolski, Filip; Dhariwal, Prafulla; Radford, Alec; Klimov, Oleg, Proximal Policy Optimization Algorithms, OpenAI, 28 August 2017, 
    https://arxiv.org/abs/1707.06347
[2] Proximal Policy Optimization, OpenAI, https://openai.com/research/openai-baselines-ppo
[3] Reinforcement Learning (PPO) with torchRL Tutorial, pytorch.org, https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
"""