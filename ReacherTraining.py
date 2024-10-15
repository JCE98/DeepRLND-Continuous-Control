import argparse, warnings, torch, time
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
    parser.add_argument("--training_episodes", type=int, nargs='?', default=2000, help="number of episodes to train agents")
    parser.add_argument("--max_iterations", type=int, nargs='?', default=1000, help="maximum number of iterations to run the episode")
    #parser.add_argument("--buffer_size", type=int, nargs='?', default=int(1e6), help="buffer size for experience replay")
    parser.add_argument("--learning_rate", type=float, nargs='?', default=1E-5, help="learning rate for agent optimization")
    parser.add_argument("--trajectory_segment", type=int, nargs='?', default=100)
    parser.add_argument("--epsilon_clip", type=float, nargs='?', default=0.1, help="clipping value for optimization surrogate function")
    parser.add_argument("--optimization_epochs", type=int, nargs='?', default=10, help="# of epochs over which to optimize the surrogate function")
    parser.add_argument("--minibatch_size", type=int, nargs='?', default=32, help="minibatch size for surrogate function optimization")
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
    for episode in range(args.training_episodes):
        start = time.time()                                                             # start time for training episode completion timer
        scores = torch.zeros(num_agents)                                                   # preallocate and initialize episode scores per agent
        env_info = env.reset(train_mode=True)[brain_name]
        states = torch.tensor(env_info.vector_observations, **tensor_kwargs)            # obtain starting states from environment reset
        for t in range(args.max_iterations):
            actions = agent.act(states)                                                 # obtain action from agent, based on policy
            env_info = env.step(actions.cpu().detach().numpy())[brain_name]             # update environment based on agent actions
            next_states = torch.tensor(env_info.vector_observations, **tensor_kwargs)   # obtain next states from updated environment
            rewards = torch.tensor(env_info.rewards, **tensor_kwargs)                   # reward for taking the action from the state
            dones = torch.tensor(env_info.local_done)                                   # check for whether the environment has met exit criteria
            agent.step(states, actions, rewards, next_states, dones)                    # record trajectory points for agent optimization
            states = next_states                                                        # update states
            scores += rewards.cpu().detach()                                            # increment running score with rewards from current step
            if any(dones):                                                              # check for environment termination
                break
            if (t+1) % args.trajectory_segment == 0:                                    # fixed-length trajectory segments
                agent.step(None, None, None, None, None, optimize=True)                 # optimization (no need to provide sars data)
        scores_deque.append(torch.mean(scores).item())                                  # record mean score for episode
        scores_array = np.append(scores_array, torch.mean(scores).item())
        print('Episode {}\tAverage Score: {:.2f}\tTime: {:.2f}'.format(episode+1, torch.mean(scores).item(), (time.time()-start)), end="\r")
        if np.mean(scores_deque) >= 33 and len(scores_deque) == 100:                    # check for environment solution
            print("\nEnvironment solved in {:.d} episodes!".format(episode+1))
            torch.save(agent.actor.state_dict(), 'solution_actor.pth')                  # save actor weights and model
            torch.save(agent.critic.state_dict(), 'solution_critic.pth')                # save critic weights and model
            break                                                                       # exit training loop if environment solved
        if episode+1 == args.training_episodes:                                         # maximum number of training episodes reached
            print('\nMax training episodes reached without environment solution!')
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
    scores = ppo(env, args)                                                             # run ppo training loop
    print('Closing Unity environment...')
    env.close()                                                                         # close environment
    print('Training Complete!')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)                                       # plot average scores per episode
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('avg_score.png')
    plt.show()
