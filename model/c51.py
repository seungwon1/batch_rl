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
        z = tf.expand_dims(tf.range(self.vmin, self.vmax+self.delta, self.delta, dtype=tf.float32), axis = 0) # of shape (1, 51)
        tz = tf.expand_dims(rew, -1) + (1-tf.expand_dims(done,-1))*self.gamma*z # of shape (32, 51)
        tz = tf.clip_by_value(tz, self.vmin, self.vmax) # clip value, of shape (32, 51)
        b = (tz - self.vmin)/self.delta # of shape (32, 51)
        l, u = tf.floor(b), tf.ceil(b) # each of shape (32, 51)
         
        q_target_ml = q_target * (u-b)
        q_target_mu = q_target * (b-l)
        
        # vectorizing l,u, target(prob) to (mini_batch, num_heads, num_heads).
        l_vec = tf.transpose(tf.expand_dims(l, -1) * tf.ones([self.mini_batch, self.num_heads, self.num_heads]), perm = [0, 2, 1]) # (32, 51, 51) 
        u_vec = tf.transpose(tf.expand_dims(u, -1) * tf.ones([self.mini_batch, self.num_heads, self.num_heads]), perm = [0, 2, 1]) # (32, 51, 51) 
        qtarget_vec_ml = tf.transpose(tf.expand_dims(q_target_ml,-1)* tf.ones([self.mini_batch, self.num_heads, self.num_heads]), perm = [0,2,1]) 
        qtarget_vec_mu = tf.transpose(tf.expand_dims(q_target_mu,-1)* tf.ones([self.mini_batch, self.num_heads, self.num_heads]), perm = [0,2,1]) 
        idx_arr = tf.expand_dims(tf.expand_dims(tf.range(self.num_heads, dtype=tf.float32), 0), -1) #  of shape (1, 51, 1)
        
        # Compute projected probability matrix
        prob_ml = tf.reduce_sum(qtarget_vec_ml * tf.cast(tf.equal(l_vec, idx_arr), tf.float32), axis = 2) # (32, 51)
        prob_mu = tf.reduce_sum(qtarget_vec_mu * tf.cast(tf.equal(u_vec, idx_arr), tf.float32), axis = 2) # (32, 51)
        m = prob_ml + prob_mu # (32, 51)
        return m
    
    # define optimizer
    def dqn_optimizer(self, loss, optim_args):
        lr = optim_args['lr']
        optimizer = tf.train.AdamOptimizer(learning_rate= lr, epsilon=0.01/32) # self.lf ,# eps1e-4 for fast convergence, 0.01/32 for QR dqn
        return optimizer.minimize(loss)
    