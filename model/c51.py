import numpy as np
import tensorflow as tf
import gym
from utils import *
from .basemodel import DQN

class C51(DQN):
    def __init__(self, num_actions, lr, mini_batch, opt, clipping, arch, gamma, vmax = 10, vmin = -10, num_heads=51):
        
        self.num_heads = num_heads
        self.vmax = vmax
        self.vmin = vmin
        super(C51, self).__init__(num_actions, lr, mini_batch, opt, clipping, arch, gamma)
        
    def c51_model(self, network):
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
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions * self.num_heads, activation_fn=None) # of shape (N, num_ac*num*heads)
            
            out = tf.reshape(out, (tf.shape(out)[0], self.num_actions, self.num_heads)) # of shape (N, num_actions, num_heads)
            out_softmax = tf.exp(out) / tf.reduce_sum(tf.exp(out), axis = 2) # apply softmax
            
            support_atoms = tf.reshape(tf.range(-vmin, vmax+1, (vmax-vmin)/(num_heads-1)), [-1, 1]) # support atoms zi, of shape (num_heads, 1)
            mean_qsa = tf.reshape(out_softmax, [-1, self.num_heads])
            mean_qsa = tf.reshape(tf.matmul(mean_qsa, support_atoms), [-1, self.num_actions]) # of shape (N, num_actions)
            greedy_idx = tf.argmax(mean_qsa, axis = 1) # of shape (N, )
            greedy_action_value = tf.reduce_max(mean_qsa, axis = 1) # of shape (N, )

        if network == 'online':
            action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # of shape (N, num_actions, 1)
            est_q = tf.matmul(out, action_mask) # of shape (N, num_actions, num_heads)
            est_q = tf.reduce_sum(out, axis = 1)  # of shape (N, num_heads)
            return state, action, q_val, est_q, batch_size, greedy_idx, greedy_action_value
        
        elif network == 'target':
            return state, greedy_action_value
        
        else:
            print('inappropriate network')
            raise
            
    def c51_loss(): # define cross-entropy loss function
        
        
    
    
    
    