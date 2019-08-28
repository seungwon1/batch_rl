import numpy as np
import tensorflow as tf
import gym
from utils import *

class DQN(object):
                                                                                                                        
    def __init__(self, num_actions, num_atoms = 51, lr = 0.00025, mini_batch = 32, opt = 'rmsprop', clipping = True, arch = "DQN", seed = 6555):

        self.num_actions = num_actions
        self.lr = lr
        self.mini_batch = mini_batch
        self.opt = opt
        self.clipping = clipping
        self.arch = arch

    # define neural networks architecture
    def model(self, network):
        state = tf.placeholder(tf.float32, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        q_val =  tf.placeholder(tf.float32, [None])
        batch_size = tf.placeholder(tf.int32, shape = None)
              
        with tf.variable_scope(network):
            with tf.variable_scope('conv'):
                conv1 = tf.contrib.layers.conv2d(state, num_outputs = 32, kernel_size = 8, stride = 4)
                conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 64, kernel_size = 4, stride = 2)
                conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 64, kernel_size = 3, stride = 1)
            conv3_flatten = tf.contrib.layers.flatten(conv3)
                
            with tf.variable_scope('fc'):
                fc1 = tf.contrib.layers.fully_connected(conv3_flatten, 512)
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions, activation_fn=None)
        
        greedy_idx = tf.argmax(out, axis = 1)
        greedy_action = tf.reduce_max(out, axis = 1)
        
        if network == 'online':
            est_q = tf.gather_nd(out, tf.transpose([tf.range(batch_size), action])) # of shape (N, )
            return state, action, q_val, est_q, batch_size, greedy_idx, greedy_action
        elif network == 'target':
            return state, greedy_action
        else:
            print('inappropriate network')
            raise
      
    # define loss function    
    def dqn_loss(self,label, pred, loss_type = 'huber'): # calculate loss
        if loss_type == 'mse':
            loss = tf.losses.mean_squared_error(label, pred)
        elif loss_type == 'huber':
            loss = tf.losses.huber_loss(label, pred)
        return loss
    
    # define optimizer
    def dqn_optimizer(self, loss):
        if self.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.lr, momentum = 0.95, epsilon = 0.01) # squared gradient?
        
        if self.clipping:  # clipping issue
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))
        else:
            train_step = optimizer().minimize(loss)
        return train_step