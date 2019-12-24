import numpy as np
import tensorflow as tf
import math
import csv
import gym
import time
import random as rand
import os
import pickle
from matplotlib import pyplot as plt
from utils import *

def make_coeff(num_heads):
    arr = np.random.uniform(low=0.0, high=1.0, size=num_heads)
    arr /= np.sum(arr)
    return arr

def eval_agent(env, sess, num_actions, gd_idx, state, FLAGS, num_games = 10):
    reward_his = []
    step_count = 0
    for j in range(num_games):
        s_t = env.reset()   
        done = False
        if step_count == 0:
            exp_memory = ex_replay(memory_size = 100)
            exp_memory.save_ex(s_t, None, None, None, None, step_count = 0)
            max_lives = env.unwrapped.ale.lives()
        
        if env.unwrapped.ale.lives() == max_lives: # accumulate rewards until agent lost all lives
            rew_epi = 0
        
        while done == False:
            # compute forward pass to find greedy action and select action with epsilon greedy strategy
            stacked_s = exp_memory.stack_frame(step_count = step_count, batch = False, b_idx = None)
            greedy_action = sess.run(gd_idx, feed_dict = {state:stacked_s})
            if rand.random() > FLAGS.eval_eps:
                a_t = greedy_action[0]
            else:
                a_t = rand.randint(0, num_actions-1)
            
            next_state, r_t, done, info = env.step(a_t)
            exp_memory.save_ex(s_t, a_t, np.sign(r_t), next_state, done, step_count = step_count)
            s_t = next_state
            rew_epi += r_t
            step_count += 1
        
        reward_his.append(rew_epi)
        step_count = 0
    mu, std = np.mean(reward_his), np.std(reward_his)
    print('Evaluation: Mean ' + str(mu) + ' std ' + str(std))
    return mu, std

class progress_stats(object):
    def __init__(self, logdir):
        self.epi_count = []
        self.loss = []
        self.reward = []
        self.avg_reward = []
        self.best_reward = []
        self.eps = []
        self.step = []
        self.total_steps = []
        self.eval_reward = []
        self.eval_reward_std = []
        self.logdir = logdir
        
    def add(self, epi_count, loss, rew, avg_rew, best_rew, eps, step, total_steps, eval_rew = None, eval_rew_std = None):
        self.epi_count.append(epi_count)
        if loss == 0:
            self.loss.append(None)
        else:
            self.loss.append(loss)
        self.reward.append(rew)
        self.avg_reward.append(avg_rew)
        self.best_reward.append(best_rew)
        self.eps.append(eps)
        self.step.append(step)
        self.total_steps.append(total_steps)
        self.eval_reward.append(eval_rew)
        self.eval_reward_std.append(eval_rew_std)
        
    def save_csv(self):
        with open(self.logdir + 'progress.csv', 'w') as f:
            f.write("episode, loss, reward, avg reward, best reward, epsilon, steps, total steps, eval_reward, eval_reward_std\n")
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip(self.epi_count, self.loss, self.reward, self.avg_reward, self.best_reward, 
                                 self.eps, self.step, self.total_steps, self.eval_reward, self.eval_reward_std))
            
 