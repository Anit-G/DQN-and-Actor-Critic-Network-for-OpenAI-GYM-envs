import random
import numpy as np
import torch.optim  as optim
import tensorflow_probability as tfp
from Qnet import QNetwork1,ReplayBuffer
import torch 
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QnetAgent():

    def __init__(self, state_size, action_size, seed = 42,
                 BUFFER_SIZE = int(1e5),
                 BATCH_SIZE = 64,
                 GAMMA = 0.99,
                 LR = 5e-4,
                 UPDATE_EVERY = 20,
                 target_fc_units = [64, 32],
                 local_fc_units = [64, 32],):
        """ Agent Hyper Paramters"""
        self.buffer_size = BUFFER_SIZE
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.lr = LR
        self.target_update = UPDATE_EVERY

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        ''' Q-Network '''
        # Double Deep Q network
        self.qnetwork_local = QNetwork1(state_size, action_size, seed, fc1_units=local_fc_units[0], fc2_units=local_fc_units[1]).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed, fc1_units=target_fc_units[0], fc2_units=target_fc_units[1]).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)
        
        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''      
        self.t_step = (self.t_step + 1) % self.target_update
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.9):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()

        # Evaluate Value function to find action value
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        ''' Epsilon-greedy action selection (Already Present) '''
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)
        # loss = F.huber_loss(Q_expected,Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()
        
        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()


from ACModel import ActorCriticModel
import tensorflow as tf

def get_expected_return(returns,discount=0.99):
    returns = returns[::-1]
    reward = returns[0]
    for t in range(1,len(returns)):
        reward = returns[t]+discount*reward
    return reward

class ACAgent:
    """
    Agent class for Actor Critic Model
    """
    def __init__(self, state_size, action_size, 
                 lr=0.001, 
                 gamma=0.99, 
                 seed = 42,
                 n_h1 = 1024,
                 n_h2 = 512,
                 return_type=1):
        
        self.gamma = gamma
        self.ac_model = ActorCriticModel(state_size=state_size, action_size=action_size,n_hidden1=n_h1, n_hidden2=n_h2)
        self.ac_model.compile(tf.keras.optimizers.Adam(learning_rate=lr))
        np.random.seed(seed)
        self.rt = return_type       # one-step, Full or n-step return

    def sample_action(self, state):
        """
        Given a state, compute the policy distribution over all actions and sample one action
        """
        pi,_ = self.ac_model(state)
        # pi,_ = self.ac_model(state[0])

        action_probabilities = tfp.distributions.Categorical(probs=pi)
        sample = action_probabilities.sample()

        return int(sample.numpy()[0])

    def actor_loss(self, action, pi, delta):
        """
        Compute Actor Loss
        """
        return -tf.math.log(pi[0,action]) * delta

    def critic_loss(self,delta):
        """
        Critic loss aims to minimize TD error
        """
        return delta**2

    @tf.function
    def learn(self, state, action, rewards, next_state, done):
        """
        For a given transition (s,a,s',r) update the paramters by computing the
        gradient of the total loss
        """
        return_type = self.rt
        with tf.GradientTape(persistent=True) as tape:
            pi, V_s = self.ac_model(state)
            _, V_s_next = self.ac_model(next_state)

            V_s = tf.squeeze(V_s)
            V_s_next = tf.squeeze(V_s_next)
            

            #### TO DO: Write the equation for delta (TD error)
            ## Write code below
            if return_type == 1 or (not isinstance(rewards,list)):
                # 1 Step Return
                delta = rewards + self.gamma*V_s_next-V_s
            elif return_type == -1:
                # Full Return, MC return
                delta = get_expected_return(rewards,self.gamma) - V_s
                pass
            elif isinstance(rewards,list) and return_type > 1:
                # N-step return
                delta = get_expected_return(rewards[:return_type],self.gamma) + self.gamma**return_type*V_s_next-V_s
                pass
            else:
                print('Incorrect Return Type or Reward structure')
                print(f"Return Type: {return_type}")
                print(f"Reward Shape: {len(rewards)}")

        
            loss_a = self.actor_loss(action, pi, delta)
            loss_c =self.critic_loss(delta)
            loss_total = loss_a + loss_c

        gradient = tape.gradient(loss_total, self.ac_model.trainable_variables)
        self.ac_model.optimizer.apply_gradients(zip(gradient, self.ac_model.trainable_variables))