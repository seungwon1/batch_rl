import numpy as np
import tensorflow as tf
from .dqn import DQN

class C51(DQN):
    
    def __init__(self, num_actions, lr = 0.00025, mini_batch = 32, opt = 'adam', clipping = False, arch = 'C51', gamma = 0.99, vmax = 10.0, vmin = -10.0, num_heads=51):
        
        super(C51, self).__init__(num_actions, lr, mini_batch, opt, clipping, arch, gamma)
        self.vmax = vmax
        self.vmin = vmin
        self.num_heads = num_heads
        self.delta = (self.vmax-self.vmin)/(self.num_heads-1)
        
    def model(self, network):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        
        batch_ns = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        batch_rew = tf.placeholder(tf.float32, [None])
        batch_done = tf.placeholder(tf.float32, [None])
        
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
            out_softmax = tf.nn.softmax(out, axis = 2) # apply softmax, of shape (N, num_actions, num_heads)

            support_atoms = tf.reshape(tf.range(self.vmin, self.vmax+self.delta, self.delta, dtype=tf.float64), [-1, 1]) 
            support_atoms = tf.cast(support_atoms, tf.float32) # support atoms 'value' zi, shape (num_heads, 1)
            
            mean_qsa = tf.reshape(out_softmax, [-1, self.num_heads]) # of shape (N*num_actions, num_heads)
            mean_qsa = tf.reshape(tf.matmul(mean_qsa, support_atoms), [-1, self.num_actions]) # of shape (N*num_actions, 1) to (N, num_actions)
            greedy_idx = tf.argmax(mean_qsa, axis = 1) # of shape (N, )
            
            if network == 'online':
                action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) #of shape (N, num_act,1)
            elif network == 'target':
                action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # (N, num_act, 1)
            
            est_q = out_softmax * action_mask # of shape (N, num_actions, num_heads)
            est_q = tf.reduce_sum(est_q, axis = 1)  # of shape (N, num_heads) # probability distribution of Q(s,a)
            raw_est_q = tf.reduce_sum(out * action_mask, axis = 1) # unnormalized output, for loss function
            
        if network == 'online':
            return state, action, est_q, greedy_idx, raw_est_q # q_val 
        
        elif network == 'target': # same as network
            project_prob = self.categorical_algorithm(est_q, batch_rew, batch_done)
            return batch_ns, batch_rew, batch_done, project_prob, est_q

    def categorical_algorithm(self, q_target, rew, done): # rew done of shape (32, ), q_target of shape (32, 51)
        m0 = tf.zeros([self.mini_batch, self.num_heads], tf.float32)  # of shape (32, 51)
        m1 = tf.ones([self.mini_batch, self.num_heads], tf.float32)  # of shape (32, 51)
        for j in range(self.num_heads):
            zj = tf.ones([self.mini_batch,], tf.float32) * (self.vmin + j*self.delta) # of shape (32,)
            tzj_1 = (1-done) * (rew + self.gamma*zj) # of shape (32,)
            tzj_2 = (done)*rew # of shape (32,)
            tzj = tf.clip_by_value(tzj_1, self.vmin, self.vmax) + tf.clip_by_value(tzj_2, self.vmin, self.vmax) # of shape (32,)
            bj = (tzj - self.vmin)/self.delta # of shape (32,)
            l, u = tf.floor(bj), tf.ceil(bj)  # of shape (32,)
            idx_l, idx_u = tf.cast(l, tf.int32), tf.cast(u, tf.int32) # for indexing
            
            ml = tf.reduce_sum(q_target*tf.one_hot(j*tf.ones([self.mini_batch,],tf.int32),self.num_heads, dtype=tf.float32), axis = 1)*(u-bj) # (32,)
            mu = tf.reduce_sum(q_target*tf.one_hot(j*tf.ones([self.mini_batch,],tf.int32),self.num_heads, dtype=tf.float32), axis = 1)*(bj-l) # (32,)
            
            m0 += (m1 * tf.one_hot(idx_l, self.num_heads, dtype=tf.float32)) * tf.reshape(ml, [-1, 1]) # of shape (32, 51)
            m0 += (m1 * tf.one_hot(idx_u, self.num_heads, dtype=tf.float32)) * tf.reshape(mu, [-1, 1]) # of shape (32, 51)
        return m0
                                                
    def c51_loss(self, project_prob, pred_prob): # Cross-entropy loss, project_prob is distribution after Bellman update
        loss = tf.reduce_mean(-tf.reduce_sum(tf.stop_gradient(project_prob) * tf.log(pred_prob+1e-7), axis = 1))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.stop_gradient(project_prob), logits = pred_prob)) # for numerical stability 
        return loss