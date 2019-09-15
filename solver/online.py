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
    
    def __init__(self, env, train_step, lr, loss, action_space, var_online, var_target, sess, pkg1, pkg2, FLAGS):
        
        self.env = env
        self.train_step = train_step
        self.lr = lr
        self.mean_loss = loss
        self.num_actions = action_space
        self.var_online = var_online
        self.var_target = var_target
        self.sess = sess
        self.state, self.action, self_est_q, self.gd_idx, _  = pkg1
        self.batch_state, self.batch_reward, self.batch_done, _, _ = pkg2
        self.FLAGS = FLAGS
        
    def train(self):
        step_count, episode_count = 0, 0
        exp_memory = ex_replay(memory_size = self.FLAGS.replay_size, batch_size = self.FLAGS.batch_size) # initialize experience replay
        loss_his, reward_his, eval_his, mean_reward, step_his, global_avg_reward = [], [], [], [], [], 0
        eps = self.FLAGS.eps
        learning_rate = self.FLAGS.lr
        time1 = time.time()
        saver = tf.train.Saver()
        # reload variable evaluate agent
        if self.FLAGS.reload or self.FLAGS.evaluate:
            exp_memory, loss_his, reward_his, step_count, mean_reward, step_count, episode_count, sess = reload_session(sess, saver, exp_memory)
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
                if step_count > self.FLAGS.train_start and step_count%self.FLAGS.update_freq == 0:
                    # load training data from memory with mini_batch size(=32)
                    batch_s, batch_a, batch_r, batch_ns, batch_done = exp_memory.sample_ex(step_count)
                    
                   # perform gradient descent to online variable
                    _, loss = self.sess.run([self.train_step, self.mean_loss], feed_dict = {self.state:batch_s, self.action:batch_a,
                                                                                            self.lr:learning_rate, self.batch_state:batch_ns,
                                                                                            self.batch_reward:batch_r,self.batch_done:batch_done})
                    loss_epi += loss
                    
                    # linearly decaying epsilon, (learning rate) for every 4th step
                    eps = linear_decay(step_count, start =1, end = 0.1, frame = 1000000)
                    if self.FLAGS.fast_test:
                        if step_count > 1e+6:
                            eps = linear_decay(step_count - 1e+6, start = 0.1, end = 0.01, frame = self.FLAGS.max_frames/2 - 1e+6)
                        if step_count >= self.FLAGS.max_frames/5 + 1:
                            learning_rate = linear_decay(step_count - self.FLAGS.max_frames/5 + 1, start = 1e-4,
                                                         end = 5e-5, frame = 0.4*self.FLAGS.max_frames)
                    
                # Reset target_variables in every interval(target_reset)
                if step_count % self.FLAGS.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])

                # increase step count, accumulates rewards
                step_count += 1
                rew_epi += r_t

            # save loss, reward per an episode, compute average reward on previous 100 number of episodes
            loss_his.append(loss_epi)
            reward_his.append(rew_epi)
            step_his.append(step_count)
            global_avg_reward = np.mean(reward_his[-100:])
            mean_reward.append(global_avg_reward)
            best_reward = np.max(mean_reward)
            
            # print progress if verbose is True, save records
            if self.FLAGS.verbose:
                show_process(self.FLAGS, episode_count ,rew_epi, global_avg_reward, best_reward, loss_epi, eps, learning_rate, step_count, 
                             step_start, time1, reward_his, step_his, mean_reward, exp_memory, loss_his, self.sess, saver)

            # increase episode_count
            episode_count += 1
            
        return loss_his, reward_his, mean_reward, eval_his