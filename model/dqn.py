import numpy as np
import tensorflow as tf

class DQN(object):
    """Nature DQN(Minh et al 2015) implementation"""
    def __init__(self, num_actions, lr = 0.00025, opt = 'rmsprop', gamma = 0.99, arch = 'DQN'):
        self.num_actions = num_actions
        self.lr = lr
        self.opt = opt
        self.gamma = gamma
        self.arch = arch
    
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
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions, activation_fn=None)
                
        greedy_idx = tf.argmax(out, axis = 1)
        greedy_action = tf.reduce_max(out, axis = 1)
        est_q = tf.reduce_sum(tf.multiply(out, tf.one_hot(act, self.num_actions, dtype='float32')), axis = 1)
        return est_q, greedy_idx, greedy_action
    
    # define huber loss
    def huber_loss(self, err, delta = 1.0):
        loss = tf.reduce_mean(tf.where(tf.abs(err) < delta, tf.square(err) * 0.5, delta * (tf.abs(err) - 0.5 * delta)))
        return loss       
        
    # Define loss function
    def loss_fn(self, online_est_q, target_args):
        # Loss function includes computing Bellman error between target Q, which is reward + gamma * max(Q_{next_state}), and online Q value
        batch_reward = target_args['batch_rew']
        batch_done = target_args['batch_done']
        tar_gd_action = target_args['gd_action_value']
        
        max_q_target = batch_reward + self.gamma*(1-batch_done)*tar_gd_action # Compute Bellman update
        error = tf.stop_gradient(max_q_target) - online_est_q # block gradient from backpropagating to the target variable
        
        loss = self.huber_loss(error, delta = 1.0)
        return loss
    
    def dqn_optimizer(self, loss, optim_args):
        # Use RMS prop, error clipping done in the paper
        lr = optim_args['lr']
        optimizer = tf.train.RMSPropOptimizer(lr, epsilon = 0.01, decay = 0.95, momentum = 0.95) 
        clipped_error = tf.clip_by_value(loss, -1, 1)
        train_step = optimizer.minimize(clipped_error)
        return train_step
