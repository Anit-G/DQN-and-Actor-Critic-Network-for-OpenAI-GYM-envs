import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
tf.compat.v1.reset_default_graph()

def plot(data,comp=1):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Episode Count')
    plt.ylabel('Reward per episode')
    plt.title(f'Reward Curve\nCompletion in {comp} episodes')
    plt.show()

def DQN(env, agent ,n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,seed = 42):

    scores = []                 
    ''' list containing scores from each episode '''

    scores_window_printing = deque(maxlen=10) 
    ''' For printing in the graph '''
    
    scores_window= deque(maxlen=100)  
    ''' last 100 scores for checking if the avg is more than 195 '''

    eps = eps_start                    
    ''' initialize epsilon '''

    for i_episode in range(1, n_episodes+1):
        try:
            state,_ = env.reset(seed=seed)
        except:
            state = env.reset(seed=seed)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            try:
                next_state, reward, done, _ , _= env.step(action)
            except:
                next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # print(reward)
            if done:
                break 

        scores_window.append(score)       
        scores_window_printing.append(score)   
        ''' save most recent score '''           

        eps = max(eps_end, eps_decay*eps) 
        ''' decrease epsilon '''

        '''Printing Scores Periodically'''
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")  
        if i_episode % 10 == 0: 
            # append to average score every 10 windows
            scores.append(np.mean(scores_window_printing))        
        if i_episode % 100 == 0: 
           # print average score every 100 windows
           print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
           print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
           break
    return [np.array(scores),i_episode-100]


def AC(env,agent, episodes=1800, num_steps=500, seed = 42):  
    average_reward_list = []

    for ep in range(1, episodes + 1):
        try:
            state,_ = env.reset(seed=seed)
        except:
            state = env.reset(seed=seed)
        state = state.reshape(1,-1)
        done = False
        ep_rew = 0
        reward_list = []
        for step in range(num_steps):
            action = agent.sample_action(state)                     ## Sample Action
            try:                                                    ## Take action
                next_state, reward, done, info, _ = env.step(action)
            except:
                next_state, reward, done, info = env.step(action)   
            next_state = next_state.reshape(1,-1)
            ep_rew += reward                                        ## Updating episode reward
            agent.learn(state, action, reward, next_state, done)    ## Update Parameters
            state = next_state                                      ## Updating State
            reward_list.append(reward)
            if done or step == num_steps-1:
                # Expected Return for full step or n-step
                agent.learn(state, action, reward_list, next_state, done) ##Update Parameters
                break

        average_reward_list.append(ep_rew)

        if ep % 10 == 0:
            avg_rew = np.mean(reward_list[-10:])
            print('Episode ', ep, 'Reward %f' % ep_rew, 'Average Reward %f' % avg_rew)

        if ep % 100:
            avg_100 =  np.mean(reward_list[-100:])
            if avg_100 > 195.0:
                print('Stopped at Episode ',ep-100)
                break
    
    return average_reward_list
        

import numpy as np
import datetime

from Agent import ACAgent,QnetAgent
from Envs import cartpole,mountaincar,acrobot
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Experiments():
    def __init__(self,batchsize=64,buffersize=int(1e5),q_gamma=0.99,q_lr=5e-4,update=20,target_units=[64,32],local_units=[64,32],
                      q_episodes=5000,max_t=1000,eps_s=1.0,eps_e=0.01,eps_d=0.995,
                      ac_lr=1e-4,ac_gamma=0.99,nh1=1024,nh2=512,rt=1,
                      ac_episodes=1800,num_steps=500):

        # DQN Agent Parameters
        self.batchsize=batchsize,
        self.buffersize=buffersize,
        self.q_gamma=q_gamma,
        self.q_lr=q_lr,
        self.update=update,
        self.target_units=target_units,
        self.local_units=local_units,

        # DQN model Parameters
        self.q_episodes=q_episodes,
        self.max_t=max_t,
        self.eps_s=eps_s,
        self.eps_e=eps_e,
        self.eps_d=eps_d,


        # Actor-Critic Agent Parameters
        self.ac_lr=ac_lr,
        self.ac_gamma=ac_gamma,
        self.nh1=nh1,
        self.nh2=nh2,
        self.rt=rt,

        # AC Model Parameters
        self.ac_episodes=ac_episodes,
        self.num_steps=num_steps
        pass

    def DQN_experiment(self,env_fn=mountaincar):
        begin_time = datetime.datetime.now()
        env, state_shape, action_shape = env_fn()
        agent = QnetAgent(state_size=state_shape, action_size = action_shape, seed = 42,
                        BUFFER_SIZE = int(1e5),
                        BATCH_SIZE = 64,
                        GAMMA = 0.99,
                        LR = 5e-4,
                        UPDATE_EVERY = 20,
                        target_fc_units = [64, 32],
                        local_fc_units = [64, 32])
        rewards_epsilon = DQN(env=env,agent=agent, 
                              n_episodes=5000, 
                              max_t=1000, 
                              eps_start=1.0,
                              eps_end=0.01,
                              eps_decay=0.995)

        time_taken = datetime.datetime.now() - begin_time
        # print(f"Time of Completion for Trainig: {time_taken}")
        return rewards_epsilon, time_taken
    
    def AC_experiment(self, env_fn=cartpole):
        begin_time = datetime.datetime.now()

        env, state_shape, action_shape = env_fn()
        agent = ACAgent(state_size = state_shape ,action_size=env.action_space.n, 
                        lr=1e-4,
                        gamma=0.99, 
                        seed = 42,
                        n_h1 = 1024,
                        n_h2 = 512,
                        return_type=1)
        rewards_epsilon = AC(env=env,agent=agent, 
                             episodes=1800, 
                             num_steps=500)

        time_taken = datetime.datetime.now() - begin_time
        # print(f"Time of Completion for Trainig: {time_taken}")
        return rewards_epsilon, time_taken
