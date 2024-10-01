# import libraries
import argparse, warnings, os
from unityagents import UnityEnvironment
import numpy as np
from PPOAgent import Agent

# command line argument parser
def checkArguments(args):
    if ((args.trajectory_segment is not None and args.episode_max is not None) and (args.trajectory_segment > args.episode_max)):
        raise Exception('Trajectory segment length provided is longer than the episode maximum number of iterations')
    if args.mode != 'training' and args.mode != 'demo':
        raise Exception("Invalid mode specified. Available modes are training and demo.")
    if args.lambd < 0 or args.lambd > 1:
        raise Exception('Lambda value must be between 0 and 1')
    if args.gamma < 0 or args.gamma > 1:
        raise Exception('Gamma value must be between 0 and 1')

def argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for PPO agent training on Reacher application")
    parser.add_argument("--mode", type=str, nargs='?',default="training", help="training or demo mode")
    parser.add_argument("--gamma", type=float, nargs='?', default=0.95, help="discount factor for future rewards, between 0 and 1")
    parser.add_argument("--lambd", type=float, nargs='?', default=0.9, help="advantage estimate discount factor, between 0 and 1")
    parser.add_argument("--training_episodes", type=int, nargs='?', default=2000, help="number of episodes to train agents")
    parser.add_argument("--episode_max", type=int, nargs='?', default=1000, help="maximum number of iterations to run the episode")
    parser.add_argument("--learning_rate", type=float, nargs='?', default=1E-5, help="learning rate for agent optimization")
    parser.add_argument("--trajectory_segment", type=int, nargs='?', default=100)
    parser.add_argument("--epsilon_clip", type=float, nargs='?', default=0.1, help="clipping value for optimization surrogate function")
    parser.add_argument("--optimization_epochs", type=int, nargs='?', default=10, help="# of epochs over which to optimize the surrogate function")
    parser.add_argument("--minibatch_size", type=int, nargs='?', default=500, help="minibatch size for surrogate function optimization")
    parser.add_argument("--actorFCunits", type=list, nargs='?', default=[64,64], help="# of neurons in each FC layer of the actor network (list)")
    parser.add_argument("--criticFCunits", type=list, nargs='?', default=[64,64], help="# of neurons in each FC layer of the critic network (list)")
    args = parser.parse_args()
    #Input processing / error handling
    checkArguments(args)
    return args

def ppo(env, args):
    # setup
    print("Performing environment setup and agent generation...")
    brain_name = env.brain_names[0]                                    # get the default brain
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]                  # reset the environment
    num_agents = len(env_info.agents)                                  # number of agents
    action_size = brain.vector_action_space_size                       # size of action space
    states = env_info.vector_observations                              # retrieve states from environment
    state_size = states.shape[1]                                       # size of the state space
    env_info = env.reset(train_mode=False)[brain_name]                 # reset the environment    
    states = env_info.vector_observations                              # get the current state (for each agent)
    agent = Agent(state_size, action_size, num_agents, args.trajectory_segment, args.learning_rate, args.actorFCunits, args.criticFCunits, seed=0)
    # training loop
    print("Entering training loop...")
    for episode in range(args.training_episodes):
        env_info = env.reset(train_mode=True)[brain_name]              # reset the environment    
        states = env_info.vector_observations                          # get the current state (for each agent)
        print(f"Episode {episode+1}:")
        iter = 1                                                       # initialize iteration counter for each episode
        epoch = 1                                                      # initialize epoch counter for each episode
        scores = np.zeros(num_agents)                                  # initialize the score (for each agent)
        while iter <= args.episode_max:                                # contain number of iterations to episode max
            print(f"\t Epoch {epoch}")
            while iter % args.trajectory_segment != 0:                 # fixed-length trajectory segments
                actions = agent.act(states)                            # select an action set for each agent
                env_info = env.step(actions)[brain_name]               # send all actions to tne environment
                next_states = env_info.vector_observations             # get next state (for each agent)
                rewards = np.reshape(np.array(env_info.rewards),(num_agents,1))                   # get reward (for each agent)
                agent.build_trajectory(states, actions, rewards, iter % args.trajectory_segment - 1) # log SAR triplets for optimization
                dones = env_info.local_done                            # see if episode finished
                scores += env_info.rewards                             # update the score (for each agent)
                states = next_states                                   # roll over states to next time step
                if np.any(dones):                                      # exit loop if episode finished
                    break
                iter+=1                                                # increment iteration counter
            if np.any(dones):                                          # exit loop if episode is finished
                break
            agent.step(args)                                           # optimize policy using trajectories (NOT YET IMPLEMENTED)
            iter+=1                                                    # increment iteration counter to move to next trajectory segment
            epoch+=1                                                   # increment epoch counter
            agent.clearBuffer()                                        # clear experience replay buffer for trajectory segment
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    print("Closing Unity environment...")
    env.close()
    print("Training Complete!")

def demoTrainedAgent(env, agentFile, n_episodes=100, max_t=200):
    #TODO: implement trained agent demonstration
    pass


if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)                              # ignore torch deprecation warnings
    # command line arguments
    print('Parsing command line arguments...')
    args = argparser()
    # environment setup
    print("Setting up environment...")
    path = "C:/Users/josia/Documents/Education/Udacity_Nanodegrees/Udacity_Deep_RL_Nanodegree/Policy_Based_Methods/Project/DeepRLND-Continuous-Control/Reacher_Windows_x86_64/Reacher.exe"
    env = UnityEnvironment(file_name=path)

    # take action based on mode selection
    if args.mode=="training":
        print("Training agents using PPO algorithm...")
        ppo(env, args)
    else:
        print("Demonstrating trained agent...")
        # trained agent checkpoint file collection
        while True:
            agentFile = input("Enter the path to the trained agent checkpoint .pth file: ").replace("\\","/")
            if not os.path.exists(agentFile):                                                               # if the file does not exist
                print("The file path you entered could not be found. Please check the path and try again.\n")
            else:
                print("Loading agent checkpoint at %s..." % agentFile)
                break
        print('\nBeginning Demonstration Run of Trained Agent...')
        demoTrainedAgent(env,)
        pass

    # results post-processing
    #TODO: implement post-processing for graphical representation of training