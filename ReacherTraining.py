import argparse, warnings, torch, time, os
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import Agent
from collections import deque
from unityagents import UnityEnvironment

def argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for PPO agent training on Reacher application")
    parser.add_argument("--mode", type=str, nargs='?',default="training", help="training or demo mode")
    parser.add_argument("--gamma", type=float, nargs='?', default=0.95, help="discount factor for future rewards, between 0 and 1")
    parser.add_argument("--lambd", type=float, nargs='?', default=0.9, help="advantage estimate discount factor, between 0 and 1")
    parser.add_argument("--training_episodes", type=int, nargs='?', default=100, help="number of episodes to train agents")
    parser.add_argument("--max_iterations", type=int, nargs='?', default=int(1E6), help="maximum number of iterations to run the episode")
    #parser.add_argument("--buffer_size", type=int, nargs='?', default=int(1e6), help="buffer size for experience replay")
    parser.add_argument("--learning_rate", type=float, nargs='?', default=3E-4, help="learning rate for agent optimization")
    parser.add_argument("--trajectory_segment", type=int, nargs='?', default=1000)
    parser.add_argument("--epsilon_clip", type=float, nargs='?', default=0.1, help="clipping value for optimization surrogate function")
    parser.add_argument("--optimization_epochs", type=int, nargs='?', default=10, help="# of epochs over which to optimize the surrogate function")
    parser.add_argument("--minibatch_size", type=int, nargs='?', default=2000, help="minibatch size for surrogate function optimization")
    parser.add_argument("--actorFCunits", type=list, nargs='?', default=[64,64], help="# of neurons in each FC layer of the actor network (list)")
    parser.add_argument("--criticFCunits", type=list, nargs='?', default=[64,64], help="# of neurons in each FC layer of the critic network (list)")
    args = parser.parse_args()
    return args

def ppo(env, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")             # tensor casting device
    tensor_kwargs = {'device':device, 'dtype':torch.float32, 'requires_grad':True}      # keyword arguments for tensor instatntiation
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    scores_deque = deque(maxlen=100)                                                    # container to capture mean scores from the last 100 episodes for exit criteria
    scores_array = np.array([])                                                         # container to capture mean scores from each episode for plotting
    agent = Agent(state_size, action_size, num_agents, args, random_seed=10)            # instantiate agent object
    for episode in range(1,args.training_episodes+1):
        start = time.time()                                                             # start time for training episode completion timer
        scores = np.zeros(num_agents)                                                   # preallocate and initialize episode scores per agent
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                                           # obtain starting states from environment reset
        for t in range(args.max_iterations):
            actions, prob_ratios = agent.act(states)                                    # obtain action from agent, based on policy
            env_info = env.step(actions)[brain_name]                                    # update environment based on agent actions
            next_states = env_info.vector_observations                                  # obtain next states from updated environment
            rewards = env_info.rewards                                                  # reward for taking the action from the state
            dones = env_info.local_done                                                 # check for whether the environment has met exit criteria
            agent.memory.add(states, actions, rewards, next_states, dones, prob_ratios) # record trajectory points for agent optimization
            states = next_states                                                        # update states
            scores += rewards                                                           # increment running score with rewards from current step
            if np.any(dones):                                                           # check for environment termination
                break
            if (t+1) % args.trajectory_segment == 0:                                    # fixed-length trajectory segments
                agent.learn()                                                           # optimization
        mean_score = np.mean(scores)
        scores_deque.append(mean_score)                                                 # record mean score for episode
        scores_array = np.append(scores_array, mean_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tTime: {:.2f}'.format(episode, mean_score, (time.time()-start)),end="")
        if episode % 100 == 0:
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            print('\nAverage Score of Last 100 episodes: {:.2f}'.format(np.mean(scores_deque)))
        if len(scores_deque) == 100 and np.mean(scores_deque) >= 33:                    # check for environment solution
            print("\nEnvironment solved in {:.d} episodes!".format(episode+1))
            torch.save(agent.actor.state_dict(), 'solution_actor.pth')                  # save actor weights and model
            torch.save(agent.critic.state_dict(), 'solution_critic.pth')                # save critic weights and model
            break                                                                       # exit training loop if environment solved
        if episode == args.training_episodes:                                           # maximum number of training episodes reached
            print('\nMax training episodes reached without environment solution!')
    return scores_array

def test_agent(env, episodes=100):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    scores_array = np.array([])                                                         # container to capture mean scores from each episode for plotting
    agent = Agent(state_size, action_size, num_agents, args, random_seed=10)            # instantiate agent object
    # Load agent solution
    agent.actor.load_state_dict(torch.load('solution_actor.pth'))
    agent.critic.load_state_dict(torch.load('solution_critic.pth'))

    scores = []

    for i_episode in range(1, episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent.reset()
        score = np.zeros(num_agents)                           # initialize the score (for each agent)

        while True:
            actions, _ = agent.act(states)                     # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            score += rewards                                   # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if any(dones):
                break

        mean_score = np.mean(score)
        scores_array = np.append(scores_array,mean_score)
        # Print out Score for each testing episode
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))

    # Compute the overall mean score from testing and print out resutls
    overall_mean = np.mean(scores)
    print('\nAverage Score {:.2f} Episodes: {:.2f}'.format(overall_mean, i_episode))
            
    return scores_array

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)                              # ignore torch deprecation warnings
    # command line arguments
    print('Parsing command line arguments...')
    args = argparser()                                                                  # parse command line hyperparameter arguments
    # environment setup
    print('Setting up environment...')
    path = "C:/Users/josia/Documents/Education/Udacity_Nanodegrees/Udacity_Deep_RL_Nanodegree/Policy_Based_Methods/Project/DeepRLND-Continuous-Control/Reacher_Windows_x86_64/Reacher.exe"
    env = UnityEnvironment(file_name=path)                                              # start Unity environment
    # train agent
    print('Entering training loop...')
    train_scores = ppo(env, args)                                                       # run ppo training loop
    print('Training Complete!')
    # plot training results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(train_scores)+1), train_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('avg_train_score.png')
    plt.show()
    # test trained agent
    if os.path.exists("./solution_actor.pth"):                                      # check whether solution has been found
        # test trained agent
        print('Testing trained agent...')
        test_scores = test_agent(env)
        print('Testing complete!')
        # plot testing results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(test_scores)+1), test_scores)                     # plot average scores per episode
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('avg_test_score.png')
        plt.show()
    print("Closing Unity environment...")
    env.close()