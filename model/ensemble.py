import numpy as np
import tensorflow as tf
from .dqn import DQN

class ENS_DQN(DQN):
    def __init__(self, num_actions, lr = 0.00025, opt = 'adam', gamma = 0.99, arch = 'ENS_DQN', num_heads=200):
        super(ENS_DQN, self).__init__(num_actions, lr, opt, gamma, arch)
        self.num_heads = num_heads

    # define neural network architecture
    def model(self, obs, act, network):
        state_float = tf.cast(obs,  tf.float32)/ 255.0
        with tf.variable_scope(network):
            with tf.variable_scope('conv'):
                conv1 = tf.contrib.layers.conv2d(state_float, num_outputs = 32, kernel_size = 8, stride = 4)
                conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 64, kernel_size = 4, stride = 2)
                conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 64, kernel_size = 3, stride = 1)
            conv3_flatten = tf.contrib.layers.flatten(conv3)

            with tf.variable_scope('fc'):
                fc1 = tf.contrib.layers.fully_connected(conv3_flatten, 512)
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions * self.num_heads, activation_fn=None) # of shape (N,num_ac*num*heads)
        
        out = tf.reshape(out, (tf.shape(out)[0], self.num_actions, self.num_heads)) # of shape (N, num_actions, num_heads)
        greedy_idx = tf.argmax(tf.reduce_mean(out, axis = 2), axis = 1) # of shape (N, )
            
        action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # of shape (N, num_act,1)
        greedy_action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1])

        est_q = tf.reduce_sum(out * action_mask, axis = 1) #of shape (N,num_act,num_heads) -> (N,num_heads) which is minimizer value(z) of Q(s,a)
        greedy_action = tf.reduce_sum(out * greedy_action_mask, axis = 1)
        return est_q, greedy_idx, greedy_action        

    # Define loss function
    def loss_fn(self, online_est_q, target_args):
        # Loss function includes computing Bellman error between target Q, which is reward + gamma * max(Q_{next_state}), and online Q value
        batch_reward = target_args['batch_rew']
        batch_done = target_args['batch_done']
        tar_gd_action = target_args['gd_action_value']
        
        max_q_target = tf.reshape(batch_reward, [-1, 1]) + self.gamma*tf.reshape((1-batch_done),[-1, 1])*tar_gd_action # Compute Bellman update
        error = tf.stop_gradient(max_q_target) - online_est_q # block gradient from backpropagating to the target variable
        
        loss = tf.reduce_mean(tf.reduce_sum(self.huber_loss(error), axis = 1))
        return loss

    # define optimizer
    def dqn_optimizer(self, loss, optim_args):
        lr = optim_args['lr']
        optimizer = tf.train.AdamOptimizer(learning_rate= lr, epsilon=0.01/32) # self.lf ,# eps1e-4 for fast convergence, 0.01/32 for QR dqn
        return optimizer.minimize(loss)
    