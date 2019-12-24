import math
import gym
import model
import time
import numpy as np
import tensorflow as tf
import random as rand
import gzip
import pickle
from utils import *
from .base import make_coeff, eval_agent, progress_stats
from .online import DQNsolver
from matplotlib import pyplot as plt

class offlineDQNsolver(DQNsolver):
    def __init__(self, env, sess, algo, FLAGS):
        super(offlineDQNsolver, self).__init__(env, sess, algo, FLAGS)
        self.tmp_memory = ex_replay(memory_size = 1000, batch_size = self.FLAGS.batch_size) # init temp memory for stacking
        
    def env_step(self, s_t, a_t, step_count): # execute action
        next_state, r_t, done, info = self.env.step(a_t)
        self.tmp_memory.save_ex(s_t, a_t, np.sign(r_t), next_state, done, step_count = step_count)
        return next_state, r_t, done, info
    
    def load_memory(self):
        with gzip.open(self.FLAGS.buffer_path, 'rb') as f:
            buffer = pickle.load(f)
        
        self.exp_memory.memory_frame = buffer[:,0]
        self.exp_memory.memory_a_r[:, 0] = buffer[:,1]
        self.exp_memory.memory_a_r[:, 1] = buffer[:,2]
        self.exp_memory.memory_a_r[:, 2] = buffer[:,3]
    
    def train(self):
        self.construct_graph()
        self.load_memory()
        self.sess.run(tf.global_variables_initializer())    
        self.sess.run([tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])
        
        # log loss and rewards
        self.tf_loss_summary = tf.summary.scalar('loss', self.loss)
        avg_reward_log = tf.placeholder(dtype=tf.float32)
        eval_avg_reward_log = tf.placeholder(dtype=tf.float32)
        tf_rew_summary = tf.summary.scalar('average reward', avg_reward_log)
        tf_eval_rew_summary = tf.summary.scalar('evaluation average reward', eval_avg_reward_log)
        tf_writer = tf.summary.FileWriter(self.FLAGS.logdir, self.sess.graph)
        stats = progress_stats(self.FLAGS.logdir)
        
        # train DQN
        reward_his, mean_reward, mean_reward, step_count, episode_count = [], [], [], 0, 0
        eps, learning_rate = self.FLAGS.eps, self.FLAGS.lr
        t_start = time.time()
        
        # train DQN
        while step_count < self.FLAGS.max_frames:
            done = False
            step_start = step_count
            s_t = self.env.reset()
            if step_count == 0: # save the first frame in the first step, assign max_lives
                self.tmp_memory.save_ex(s_t, None, None, None, None, step_count = step_count)
                max_lives = self.env.unwrapped.ale.lives()
                
            if self.env.unwrapped.ale.lives() == max_lives: # accumulate rewards until agent lost all lives
                rew_epi, loss_epi = 0, 0
                
            while done == False: # continue to step until an episode terminates
                stacked_s = self.tmp_memory.stack_frame(step_count = step_count, batch = False, b_idx = None)
                a_t = self.eps_greedy(stacked_s, eps)
                s_t, r_t, done, info = self.env_step(s_t, a_t, step_count)
                loss, loss_val = self.perform_gd(step_count)
                if step_count % self.FLAGS.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])
                
                if loss is not None:
                    tf_writer.add_summary(loss, step_count)
                    loss_epi += loss
                rew_epi += r_t
                step_count += 1
                
            if self.env.unwrapped.ale.lives() == 0:
                avg_rew = self.sess.run(tf_rew_summary, feed_dict={avg_reward_log:rew_epi})
                tf_writer.add_summary(avg_rew, step_count)
                
                # save loss, reward per an episode, compute average reward on previous 100 number of episodes
                reward_his.append(rew_epi)
                avg_reward = np.mean(reward_his[-100:])
                mean_reward.append(avg_reward)
            
                # Print progress
                if self.FLAGS.verbose and episode_count % self.FLAGS.print_every == 0:
                    t_inter = time.time()
                    print('\nEpi {0}: rew {1:2g}, avg_r {2:2g} best_r {3:2g}, eps {4:2g}, lr {5:2g}, steps {6}, t_steps {7}, wall clock min {8:2g}'\
                          .format(episode_count ,rew_epi, avg_reward, np.max(mean_reward), eps, learning_rate,  step_count - step_start,
                                  step_count, (t_inter - t_start)/60))
                   
                # evaluate agent, log progress
                if self.FLAGS.evaluate and episode_count % self.FLAGS.eval_every == 0:
                    eval_rew, eval_std = eval_agent(self.env, self.sess, self.num_actions, self.on_gd_idx, self.on_state, self.FLAGS)
                    e_rew = self.sess.run(tf_eval_rew_summary, feed_dict={eval_avg_reward_log:eval_rew})
                    tf_writer.add_summary(e_rew, step_count)
                    stats.add(episode_count, loss_epi/(step_count-step_start), rew_epi, avg_reward,
                              np.max(mean_reward), eps, step_count - step_start, step_count, eval_rew, eval_std)
                else:
                    stats.add(episode_count, loss_epi/(step_count-step_start), rew_epi,
                          avg_reward, np.max(mean_reward), eps, step_count - step_start, step_count)
                
                stats.save_csv()
                episode_count += 1
