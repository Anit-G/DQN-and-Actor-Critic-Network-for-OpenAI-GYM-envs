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


def DQN(env, agent ,n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    scores = []                 
    ''' list containing scores from each episode '''

    scores_window_printing = deque(maxlen=10) 
    ''' For printing in the graph '''
    
    scores_window= deque(maxlen=100)  
    ''' last 100 scores for checking if the avg is more than 195 '''

    eps = eps_start                    
    ''' initialize epsilon '''

    for i_episode in range(1, n_episodes+1):
        state,_ = env.reset(seed=0)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ , _= env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            print(reward)
            if done:
                break 

        scores_window.append(score)       
        scores_window_printing.append(score)   
        ''' save most recent score '''           

        eps = max(eps_end, eps_decay*eps) 
        ''' decrease epsilon '''

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


def AC(env,agent,episodes=1800,num_steps=500):  
    average_reward_list = []

    for ep in range(1, episodes + 1):
        state,_ = env.reset(seed=0)
        state = state.reshape(1,-1)
        done = False
        ep_rew = 0
        reward_list = []
        for step in range(num_steps):
            action = agent.sample_action(state) ##Sample Action
            next_state, reward, done, info, _ = env.step(action) ##Take action
            next_state = next_state.reshape(1,-1)
            ep_rew += reward  ##Updating episode reward
            reward_list.append(reward)
            agent.learn(state, action, reward, next_state, done) ##Update Parameters
            state = next_state ##Updating State

            if done or step == num_steps-1:
                # Expected Return for full step or n-step
                agent.learn(state, action, reward_list, next_state, done) ##Update Parameters

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
        