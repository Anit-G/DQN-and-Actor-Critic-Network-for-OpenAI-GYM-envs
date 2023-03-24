import tensorflow as tf

class ActorCriticModel(tf.keras.Model):
    """
    Defining policy and value networkss
    """
    def __init__(self,state_size ,action_size, n_hidden1=1024, n_hidden2=512, seed = 42):
        super(ActorCriticModel, self).__init__()
        tf.random.set_seed(seed=seed)
        # Input layer
        # self.input_layer = tf.keras.layers.Input(state_size)
        #Hidden Layer 1
        self.fc1 = tf.keras.layers.Dense(n_hidden1, activation='relu')
        #Hidden Layer 2
        self.fc2 = tf.keras.layers.Dense(n_hidden2, activation='relu')
        
        #Output Layer for policy
        self.pi_out = tf.keras.layers.Dense(action_size, activation='softmax')
        #Output Layer for state-value
        self.v_out = tf.keras.layers.Dense(1)

    def call(self, state):
        """
        Computes policy distribution and state-value for a given state
        """
        # state = self.input_layer(state)
        layer1 = self.fc1(state)
        layer2 = self.fc2(layer1)

        pi = self.pi_out(layer2)
        v = self.v_out(layer2)

        return pi, v
    
import torch
import torch.nn.functional as F

class ActorCritic_Torch(torch.nn.Module):
    def __init__(self,state_size, action_size, hidden1_dim, hidden2_dim):
        super(ActorCritic_Torch, self).__init__()
        self.fc1 = torch.nn.Linear(state_size,hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim,hidden2_dim)
        self.actor = torch.nn.Linear(hidden2_dim,action_size)
        self.critic = torch.nn.Linear(hidden2_dim,1)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = F.softmax(self.actor(x))
        # pi = F.tanh(self.actor(x))
        v = self.critic(x)

        return pi,v

