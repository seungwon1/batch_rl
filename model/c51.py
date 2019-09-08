import numpy as np
import tensorflow as tf
import gym
from utils import *
from .basemodel import DQN

class C51(DQN):
    def __init__(self, num_actions, lr, opt, clipping, arch, gamma = 0.99, mini_batch = 32, vmax = 10, vmin = -10, num_heads=51):
        
        self.num_heads = num_heads
        self.vmax = vmax
        self.vmin = vmin
        super(C51, self).__init__(num_actions, lr, opt, clipping, arch, gamma, mini_batch)
        
    def c51_model(self, network):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        q_val =  tf.placeholder(tf.float32, [None])

        batch_ns = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        batch_rew = tf.placeholder(tf.float32, [None, 1])
        batch_done_idx = tf.placeholder(tf.int32, [None, 1])
        
        if network == 'online':
            state_float = tf.cast(state,  tf.float32)/ 255.0
        elif network == 'target':
            state_float = tf.cast(batch_ns,  tf.float32)/ 255.0
        
        with tf.variable_scope(network):
            with tf.variable_scope('conv'):
                conv1 = tf.contrib.layers.conv2d(state_float, num_outputs = 32, kernel_size = 8, stride = 4)
                conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 64, kernel_size = 4, stride = 2)
                conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 64, kernel_size = 3, stride = 1)
                
            conv3_flatten = tf.contrib.layers.flatten(conv3)
                
            with tf.variable_scope('fc'):
                fc1 = tf.contrib.layers.fully_connected(conv3_flatten, 512)
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions * self.num_heads, activation_fn=None) # of shape (N, num_ac*num*heads)
            
            out = tf.reshape(out, (tf.shape(out)[0], self.num_actions, self.num_heads)) # of shape (N, num_actions, num_heads)
            out_softmax = tf.nn.softmax(out, axis = 2) # apply softmax, of shape (N, num_actions, num_heads)
            
            support_atoms = tf.reshape(tf.range(-vmin, vmax+1, (vmax-vmin)/(num_heads-1)), [-1, 1]) # support atoms 'value' zi, shape (num_heads, 1)
            mean_qsa = tf.reshape(out_softmax, [-1, self.num_heads])
            mean_qsa = tf.reshape(tf.matmul(mean_qsa, support_atoms), [-1, self.num_actions]) # of shape (N, num_actions)
            greedy_idx = tf.argmax(mean_qsa, axis = 1) # of shape (N, )
            # greedy_action_value = tf.reduce_max(mean_qsa, axis = 1) # of shape (N, )
            
            if network == 'online':
                action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # of shape (N, num_act, 1)
            elif network == 'target':
                action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # (N, num_act, 1)

            est_q = tf.matmul(out_softmax, action_mask) # of shape (N, num_actions, num_heads)
            est_q = tf.reduce_sum(out, axis = 1)  # of shape (N, num_heads) # probability distribution of Q(s,a)

        if network == 'online':
            return state, action, est_q, greedy_idx  # q_val 
        
        elif network == 'target': # same as network
            update_support = batch_rew + self.gamma * tf.matmul((1-batch_done_idx), tf.reshape(support_atoms, [1, -1])) # of shape (N, num_heads)
            return batch_ns, update_support, est_q
        
        else:
            print('inappropriate network')
            raise
                                           
    def categorical_algorithm(est_q_online, est_q_target, update_support):
        for i in range(self.num_heads):
            zi_prob = tf.reduce_sum(tf.multiplay(est_q_online, tf.one_hot(i*np.ones((sel.fnum_heads,)), 2, dtype = 'float32')), axis = 1)
            project_zi_prob = tf.reduce_sum(tf.clip_by_value((1 - tf.abs(tf.clip_by_value(update_support, self.vmin, self.vmax) - zi)/((vmax-vmin)/(num_heads-1))), 0, 1) * est_q, axis = 1, keepdims = True) # of shape (batch_size,1)
            if i == 0:
                prob_logits = project_zi_prob
            else:
                prob_logits = tf.concat([prob_temp, project_zi_prob])
        return prob_logits
                                                
    def c51_loss(self,logit_prob, pred_prob, loss_type = 'huber'): # Cross-entropy loss, logit_prob is distribution after Bellman update
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = pred_prob, logits = logit_prob)) # for numerical stability 
        return loss
