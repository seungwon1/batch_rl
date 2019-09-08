import numpy as np
import tensorflow as tf
import math
import gym
from utils import *
from .base import e_greedy_execute, env_step, show_process
import time
from matplotlib import pyplot as plt
from PIL import Image

class dqn_online_solver(object):
    
    def __init__(self, env, train_step, loss, action_space, var_online, var_target, sess, pkg1, pkg2, FLAGS):
        
        self.env = env
        self.train_step = train_step
        self.mean_loss = loss
        self.num_actions = action_space
        self.var_online = var_online
        self.var_target = var_target
        self.sess = sess
        self.state, self.action, self.q_val, self_est_q, self.gd_idx, self.gd_action  = pkg1
        self.state_target, self.max_q_target = pkg2
        self.FLAGS = FLAGS
        
    def train(self):
        step_count, episode_count = 0, 0
        exp_memory = ex_replay(memory_size = self.FLAGS.replay_size, batch_size = self.FLAGS.batch_size) # initialize experience replay
        loss_his, reward_his, eval_his, mean_reward = [], [], [], []
        global_avg_reward = 0
        eps = self.FLAGS.eps
        time1 = time.time()
        saver = tf.train.Saver()
        # reload variable evaluate agent
        if self.FLAGS.reload or self.FLAGS.evaluate:
            exp_memory, loss_his, reward_his, step_count = reload_session(saver)
            # evaluate agent
            if self.FLAGS.evaluate:
                eval_rew_his = eval_agent(self.num_games, self.env, exp_memory, self.sess, self.num_actions, self.gd_idx, state. self.FLAGS)
                return eval_new_his, None, None, None
            
        while step_count < self.FLAGS.max_frames:
            rew_epi, loss_epi = 0, 0   
            done = False
            step_start = step_count
            s_t = self.env.reset()   
            if step_count == 0: # only save first frame
                exp_memory.save_ex(s_t, None, None, None, None, step_count = step_count)
                
            while done == False: # continue to step until an episode terminates
                # compute forward pass to find greedy action and select action with epsilon greedy strategy
                stacked_s = exp_memory.stack_frame(b_idx = None, step_count = step_count, batch = False)
                greedy_action = self.sess.run([self.gd_idx], feed_dict = {self.state:stacked_s})
                a_t = e_greedy_execute(self.num_actions, eps, greedy_action)
                    
                next_state, r_t, done, info = self.env.step(a_t) #env_step(self.env, s_t, a_t, self.FLAGS) # step ahead. skip every kth frame  

                # save (s_t, a_t, r_t, s_t+1) to memory
                exp_memory.save_ex(s_t, a_t, r_t, next_state, done, step_count = step_count) # a_t and r_t is int and float
                s_t = next_state
                
                # if memory collects enough data, start training (perform gradient descent with respect to online variable)
                if step_count > self.FLAGS.train_start and step_count%4 == 0:
                    # load training data from memory with mini_batch size(=32)
                    batch_s, batch_a, batch_r, batch_ns, batch_done = exp_memory.sample_ex(step_count)
                    
                    # use fixed target variable to calculate target max_Q
                    max_q_hat = self.sess.run(self.max_q_target, feed_dict = {self.state_target:batch_ns})
                    target = batch_r + self.FLAGS.gamma*max_q_hat # target_q_val, of shape (minibatch, 1)
                    target[batch_done == 1] = batch_r[batch_done == 1] # assign reward only if next state is terminal

                    # perform gradient descent to online variable
                    _, loss = self.sess.run([self.train_step, self.mean_loss], feed_dict = {self.state:batch_s, self.action:batch_a,  self.q_val:target})
                    loss_epi += loss
                    # linearly decaying epsilon for every 4 step
                    eps = linear_decay(step_count)
                    
                # Reset target_variables in every interval(target_reset)
                if step_count % self.FLAGS.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])

                # increase step count, accumulates rewards
                step_count += 1
                rew_epi += r_t

            # save loss, reward per an episode, compute average reward on previous 100 number of episodes
            loss_his.append(loss_epi)
            reward_his.append(rew_epi)
            global_avg_reward = np.mean(reward_his)
            mean_reward.append(global_avg_reward)
            
            # print progress if verbose is True, save records
            if self.FLAGS.verbose:
                show_process(self.FLAGS, episode_count ,rew_epi, global_avg_reward, loss_epi, eps, step_count, 
                             step_start, time1, reward_his, mean_reward, exp_memory, loss_his, self.sess, saver)

            # increase episode_count
            episode_count += 1
            
        return loss_his, reward_his, mean_reward, eval_his