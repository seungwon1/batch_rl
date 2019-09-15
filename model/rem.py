import numpy as np
import tensorflow as tf
from .dqn import DQN

class REM_DQN(DQN):
    
    def __init__(self, num_actions, lr = 0.00005, opt = 'adam', clipping = False, arch = 'QR_DQN', gamma = 0.99, mini_batch = 32, num_heads=200):
        
        super(QR_DQN, self).__init__(num_actions, lr, mini_batch, opt, clipping, arch, gamme)
        self.num_heads = num_heads
        
    def model(self, network):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        random_coeff = tf.placeholder(tf.float32, [None]) # originally of shape (H, ). this will broadcast to (N, A, H)
        
        batch_ns = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        batch_rew = tf.placeholder(tf.float32, [None])
        batch_done_idx = tf.placeholder(tf.float32, [None])
        
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
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions * self.num_heads, activation_fn=None) # of shape (N,num_ac*num*heads)
            
            out = tf.reshape(out, (tf.shape(out)[0], self.num_actions, self.num_heads)) # of shape (N, num_actions, num_heads)
            out *= tf.reshape(random_coeff, [1,1,-1]) # same shape as above
            greedy_idx = tf.argmax(tf.reduce_sum(out, axis = 2), axis = 1) # of shape (N, )
            
            if network == 'online':
                action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) #of shape (N, num_act,1)
            elif network == 'target':
                action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # (N, num_act, 1)
            
            est_q = tf.reduce_sum(out * action_mask, axis = 1)# of shape (N,num_heads) which is random ensemble model of Q(s,a)
            est_q = tf.reduce_sum(est_q, axis = 1) # of shape (N, )
            
        if network == 'online':
            return state, action, est_q, greedy_idx, random_coeff
        
        elif network == 'target':
            update_est_q = batch_rew + self.gamma*(1-batch_done_idx)*est_q # of shape (N,)
            return batch_ns, batch_rew, batch_done, update_est_q, random_coeff
    
    def loss_fn(self, target_q, online_q, delta = 1.0): # use huber loss
        err = tf.stop_gradient(target_q) - online_q # of shape (N,)
        loss = tf.reduce_mean(self.huber_loss(err, delta))
        return loss
    