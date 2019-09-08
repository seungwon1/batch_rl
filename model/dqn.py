import numpy as np
import tensorflow as tf
import gym
from utils import *

class DQN(object):
                                                                                                                        
    def __init__(self, num_actions, num_atoms = 51, lr = 0.00025, mini_batch = 32, opt = 'rmsprop', clipping = True, arch = "DQN", gamma = 0.99):

        self.num_actions = num_actions
        self.lr = lr
        self.mini_batch = mini_batch
        self.opt = opt
        self.clipping = clipping
        self.arch = arch
        self.gamma = gamma
        
    # define neural networks architecture
    def model(self, network):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        q_val =  tf.placeholder(tf.float32, [None])
        
        state_float = tf.cast(state,  tf.float32)/ 255.0
        
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
        
        if network == 'online':
            est_q = tf.reduce_sum(tf.multiply(out, tf.one_hot(action, self.num_actions, dtype='float32')), axis = 1)
            return state, action, q_val, est_q, greedy_idx, greedy_action
        elif network == 'target':
            return state, greedy_action
        else:
            print('inappropriate network')
            raise
      
    def dqn_loss(self,label, pred, loss_type = 'huber', delta = 1.0): # calculate loss
        if loss_type == 'mse':
            loss = tf.losses.mean_squared_error(label, pred)
        elif loss_type == 'huber':
            x = label - pred
            loss = tf.reduce_mean(tf.where(tf.abs(x) < delta, tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta)))
        return loss
    
    # define optimizer
    def dqn_optimizer(self, loss, variables):
        if self.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(1e-4) # self.lf
        elif self.opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.lr) #, momentum = 0.95, epsilon = 0.01) # squared gradient?
        if self.clipping:  # clipping issue
            """
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))
            """
            gradients = optimizer.compute_gradients(loss, var_list=variables)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 10), var)
            return optimizer.apply_gradients(gradients)
        else:
            train_step = optimizer().minimize(loss)
        return train_step