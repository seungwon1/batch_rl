import math
import gym
import model
import time
import numpy as np
import tensorflow as tf
from utils import *
from .base import show_process, reload_session
from matplotlib import pyplot as plt

class DQNsolver(object):
    def __init__(self, env, sess, algo, FLAGS):
        self.env = env
        self.num_actions = env.action_space.n
        self.sess = sess 
        self.algo = algo 
        self.FLAGS = FLAGS
        self.exp_memory = ex_replay(memory_size = self.FLAGS.replay_size, batch_size = self.FLAGS.batch_size) # init ex memory
        
    def define_place_holder(self):
        state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        action = tf.placeholder(tf.int32, [None])
        
        batch_next_state = tf.placeholder(tf.uint8, [None, 84, 84, 4])
        batch_reward = tf.placeholder(tf.float32, [None])
        batch_done = tf.placeholder(tf.float32, [None])
        return state, action, batch_next_state, batch_reward, batch_done
    
    def construct_graph(self):
        self.on_state, self.on_action, _, _, _ = self.define_place_holder()
        _, self.batch_action, self.batch_ns, self.batch_reward, self.batch_done = self.define_place_holder()
        
        on_est_q, self.on_gd_idx, _ = self.algo.model(self.on_state, self.on_action, 'online')
        _, _, tar_gd_action = self.algo.model(self.batch_ns, self.batch_action, 'target')
        self.var_online = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'online')
        self.var_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'target')
        
        self.loss = self.algo.loss_fn(on_est_q, target_args = {'batch_rew': self.batch_reward,'batch_done':self.batch_done,
                                                               'gd_action_value':tar_gd_action})
        self.train_step = self.algo.dqn_optimizer(self.loss, optim_args={'lr':self.FLAGS.lr})
        
    """
    def preprocessing(self, obs):
        # Use gym wrapper instead
        pass
    """
        
    def eps_greedy(self, obs, eps):
        greedy_action = self.sess.run(self.on_gd_idx, feed_dict = {self.on_state:obs})
        indicator = np.random.choice(2, 1, p=[eps, 1-eps])
        if indicator == 1:
            a_t = greedy_action[0]
        else:
            a_t = np.random.choice(self.num_actions, 1)[0]
        return a_t
      
    def env_step(self, s_t, a_t, step_count): # execute action and save trainsition
        next_state, r_t, done, info = self.env.step(a_t)
        self.exp_memory.save_ex(s_t, a_t, r_t, next_state, done, step_count = step_count) # a_t and r_t is int and float
        return next_state, r_t, done, info
        
    def perform_gd(self, step_count): # perform gradient descent
        # if memory collects enough data, start training (perform gradient descent with respect to online variable)
        if step_count > self.FLAGS.train_start and step_count%self.FLAGS.update_freq == 0:
            # load training data from memory with mini_batch size(=32)
            batch_s, batch_a, batch_r, batch_ns, batch_done = self.exp_memory.sample_ex(step_count)
                    
            # perform gradient descent with respect to online variable
            if self.FLAGS.arch == 'REM_DQN':
                random_coeff = make_coeff(self.num_heads)
                _, loss = self.sess.run([self.train_step, self.loss], 
                                        feed_dict = {self.on_state:batch_s, self.on_action:batch_a, self.lr:learning_rate,                                                          self.batch_ns:batch_ns, self.batch_reward:batch_r,
                                                     self.batch_done:batch_done, self.rco1:random_coeff,
                                                     self.rco2:random_coeff})
            else:
                _, loss = self.sess.run([self.train_step, self.loss],
                                        feed_dict = {self.on_state:batch_s, self.on_action:batch_a, self.batch_ns:batch_ns, 
                                                     self.batch_reward:batch_r, self.batch_done:batch_done})
            return loss
    
    def train(self):
        # This function implements DQN with experience replay(Mnih et al. 2015)
        self.construct_graph()
        
        # initialize variables, assign target variables to online variables
        self.sess.run(tf.global_variables_initializer())    
        self.sess.run([tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])
        
        # train DQN
        loss_his, reward_his, mean_reward, step_his, global_avg_reward, mean_reward, step_count, episode_count = [], [], [], [], [], [], 0, 0
        eps, learning_rate = self.FLAGS.eps, self.FLAGS.lr
        
        """For reloading and evaluation"""
        time1 = time.time()
        saver = tf.train.Saver()
        # reload variable evaluate agent
        if self.FLAGS.reload or self.FLAGS.evaluate:
            self.exp_memory, loss_his, reward_his, step_count, mean_reward, episode_count, step_his, step_start, self.sess = reload_session(self.sess, saver, exp_memory, self.FLAGS)
            
            # evaluate agent
            if self.FLAGS.evaluate:
                eval_rew_his = eval_agent(self.num_games, self.env, self.exp_memory, self.sess, self.num_actions, self.gd_idx, self.on_state, self.FLAGS)
                return eval_new_his, None, None, None
        """For reloading and evaluation"""    
        
        # train DQN
        while step_count < self.FLAGS.max_frames:
            rew_epi, loss_epi = 0, 0
            done = False
            step_start = step_count
            s_t = self.env.reset()
            if step_count == 0: # save the first frame in the first step.
                self.exp_memory.save_ex(s_t, None, None, None, None, step_count = step_count)
                
            while done == False: # continue to step until an episode terminates
                # compute forward pass to find greedy action and select action with epsilon greedy strategy
                stacked_s = self.exp_memory.stack_frame(step_count = step_count, batch = False, b_idx = None)
                a_t = self.eps_greedy(stacked_s, eps)
                
                # execute action, observe reward and next state, save (s_t, a_t, r_t, s_t+1) to experience replay
                next_state, r_t, done, info = self.env_step(s_t, a_t, step_count)
                s_t = next_state
                
                # if memory collects enough data, perform a gradient descent step with respect to the online variable
                loss = self.perform_gd(step_count)
                
                # Reset target_variable in every target_reset interval
                if step_count % self.FLAGS.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])
                
                # etc (decaying epsilon, acculates loss, rewards, increase step count and episode count)
                eps = linear_decay(step_count, start =self.FLAGS.eps, end = self.FLAGS.final_eps, frame = 1000000)
                if loss is not None:
                    loss_epi += loss
                rew_epi += r_t
                step_count += 1
            episode_count += 1
            
            """For saving and printing results of experiments"""
            # save loss, reward per an episode, compute average reward on previous 100 number of episodes
            if loss_epi != 0:
                loss_his.append(loss_epi/(step_count-step_start))
            reward_his.append(rew_epi)
            step_his.append(step_count)
            global_avg_reward = np.mean(reward_his[-100:])
            mean_reward.append(global_avg_reward)
            best_reward = np.max(mean_reward)
            
            # print progress if verbose is True, save records
            if self.FLAGS.verbose:
                show_process(self.FLAGS, episode_count-1 ,rew_epi, global_avg_reward, best_reward, loss_epi, eps,
                             learning_rate, step_count, step_start, time1, reward_his, step_his, mean_reward, self.exp_memory, 
                             loss_his, self.sess, saver)
            """For saving and printing results of experiments"""
            