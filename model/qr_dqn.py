import numpy as np
import tensorflow as tf
from .dqn import DQN

class QR_DQN(DQN):
    def __init__(self, num_actions, lr = 0.00005, opt = 'adam', clipping = False, arch = 'QR_DQN', gamma = 0.99, mini_batch = 32, num_heads=200):
        
        super(QR_DQN, self).__init__(num_actions, lr, mini_batch, opt, clipping, arch, gamma)
        self.num_heads = num_heads
        
    def model(self, network):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        
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
            greedy_idx = tf.argmax(tf.reduce_sum(out, axis = 2)/self.num_heads, axis = 1) # of shape (N, )
            
            if network == 'online':
                action_mask = tf.reshape(tf.one_hot(action, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) #of shape (N, num_act,1)
            elif network == 'target':
                action_mask = tf.reshape(tf.one_hot(greedy_idx, self.num_actions, dtype='float32'), [-1, self.num_actions, 1]) # (N, num_act, 1)
            
            est_q = tf.reduce_sum(out * action_mask, axis = 1) #of shape (N,num_act,num_heads) -> (N,num_heads) which is minimizer value(z) of Q(s,a)
            est_q = tf.sort(est_q) # sort elements for calulating quantile regression loss
            
        if network == 'online':
            return state, action, est_q, greedy_idx, None 
        
        elif network == 'target':
            update_zi = tf.reshape(batch_rew, [-1, 1]) + self.gamma*tf.reshape((1-batch_done_idx),[-1, 1])* est_q # of shape (N,num_heads),T_theta
            return batch_ns, batch_rew, batch_done_idx, update_zi

    def loss_fn(self, target_q, online_q, delta = 1.0): 
        # QR huber loss, target_val is value of supports after applying Bellman update
        target_q = tf.expand_dims(target_q, axis = -2) # (N, 1, H)
        online_q = tf.expand_dims(online_q, axis = -1) # (N, H, 1)
        
        u = tf.stop_gradient(target_q) - online_q # (N, H, H)
        indicator = tf.cast(u < 0.0, tf.float32)
        quentiles = ((np.arange(self.num_heads)+0.5)/float(self.num_heads)).reshape(1,self.num_heads, 1)
        
        loss = tf.abs(quentiles - indicator) * self.huber_loss(u, delta) # (N, H, H) # tf.stop_gradient(indicator)
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis = 2), axis = 1)) 
        return loss
