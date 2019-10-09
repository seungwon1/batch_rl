import numpy as np
import tensorflow as tf
from .dqn import DQN

class C51(DQN):
    """C51 DQN(Bellemare et al 2017) implementation"""
    def __init__(self, num_actions, lr = 0.00025, opt = 'adam', gamma = 0.99, arch = 'C51', vmax = 10.0, vmin = -10.0, num_heads=51, mini_batch=32):
        super(C51, self).__init__(num_actions, lr, opt, gamma, arch)
        self.vmax = vmax
        self.vmin = vmin
        self.num_heads = num_heads
        self.delta = (self.vmax-self.vmin)/(self.num_heads-1)
        self.mini_batch = mini_batch
        
    # Define neural network architecture. Output layer outputs probability of each support
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
            out_softmax = tf.nn.softmax(out, axis = 2) # apply softmax, of shape (N, num_actions, num_heads)

            support_atoms = tf.reshape(tf.range(self.vmin, self.vmax+self.delta, self.delta, dtype=tf.float64), [-1, 1]) 
            support_atoms = tf.cast(support_atoms, tf.float32) # support atoms 'value' zi, shape (num_heads, 1)
            
            mean_qsa = tf.reshape(out_softmax, [-1, self.num_heads]) # of shape (N*num_actions, num_heads)
            mean_qsa = tf.reshape(tf.matmul(mean_qsa, support_atoms), [-1, self.num_actions]) # of shape (N*num_actions, 1) to (N, num_actions)
            greedy_idx = tf.argmax(mean_qsa, axis = 1) # of shape (N, )
            
            action_mask = tf.reshape(tf.one_hot(act, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) #of shape (N, num_act,1)
            greedy_action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # (N, num_act, 1)
            
            est_q = out_softmax * action_mask # of shape (N, num_actions, num_heads)
            est_q = tf.reduce_sum(est_q, axis = 1)  # of shape (N, num_heads) # probability distribution of Q(s,a)
            
            greedy_action = out_softmax * greedy_action_mask
            greedy_action = tf.reduce_sum(greedy_action, axis = 1)
            
        return est_q, greedy_idx, greedy_action
        
    # Define loss function
    def loss_fn(self, online_est_q, target_args): 
        # Loss function is cross-entropy loss, project_prob is distribution after Bellman update
        batch_reward = target_args['batch_rew']
        batch_done = target_args['batch_done']
        tar_gd_action = target_args['gd_action_value']
        
        project_prob = self.categorical_algorithm(tar_gd_action, batch_reward, batch_done)
        loss = tf.reduce_mean(-tf.reduce_sum(tf.stop_gradient(project_prob) * tf.log(online_est_q+1e-10), axis = 1))
        return loss
    
    def categorical_algorithm(self, q_target, rew, done): # rew, done each of shape (32, ), q_target of shape (32, 51)
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
    
    # define optimizer
    def dqn_optimizer(self, loss, optim_args):
        lr = optim_args['lr']
        optimizer = tf.train.AdamOptimizer(learning_rate= lr, epsilon=0.01/32) # self.lf ,# eps1e-4 for fast convergence, 0.01/32 for QR dqn
        return optimizer.minimize(loss)
    