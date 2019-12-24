import numpy as np
import tensorflow as tf
from .dqn import DQN

class QR_DQN(DQN):
    """QR DQN(Dabney et al 2017) implementation"""
    def __init__(self, num_actions, lr = 0.00025, opt = 'adam', gamma = 0.99, arch = 'QR_DQN', num_heads=200):
        super(QR_DQN, self).__init__(num_actions, lr, opt, gamma, arch)
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
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions * self.num_heads, activation_fn=None)
        
        out = tf.reshape(out, (tf.shape(out)[0], self.num_actions, self.num_heads))
        greedy_idx = tf.argmax(tf.reduce_sum(out, axis = 2)/self.num_heads, axis = 1)
        
        action_mask = tf.reshape(tf.one_hot(act, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) 
        greedy_action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1])

        est_q = tf.reduce_sum(out * action_mask, axis = 1)
        est_q = tf.sort(est_q)
            
        greedy_action = tf.reduce_sum(out * greedy_action_mask, axis = 1)
        greedy_action = tf.sort(greedy_action)    
        return est_q, greedy_idx, greedy_action

    # Define loss function
    def loss_fn(self, online_est_q, target_args, delta = 1.0): 
        batch_reward = target_args['batch_rew']
        batch_done = target_args['batch_done']
        tar_gd_action = target_args['gd_action_value']
        
        update_zi = tf.reshape(batch_reward, [-1, 1]) + self.gamma*tf.reshape((1-batch_done),[-1, 1])*tar_gd_action
        target_q = update_zi
        
        target_q = tf.expand_dims(target_q, axis = -2)
        online_est_q = tf.expand_dims(online_est_q, axis = -1)
        
        u = tf.stop_gradient(target_q) - online_est_q
        indicator = tf.cast(u < 0.0, tf.float32)
        quentiles = ((np.arange(self.num_heads)+0.5)/float(self.num_heads)).reshape(1,self.num_heads, 1)
        
        loss = tf.abs(quentiles - indicator) * self.huber_loss(u, delta)
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis = 2), axis = 1)) 
        return loss
