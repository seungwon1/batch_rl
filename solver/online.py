import math
import gym
import model
import time
import numpy as np
import tensorflow as tf
import random as rand
from utils import *
from .base import make_coeff, eval_agent, progress_stats
from matplotlib import pyplot as plt

class DQNsolver(object):
    def __init__(self, env, sess, algo, FLAGS):
        self.env = env
        self.num_actions = env.action_space.n
        self.sess = sess 
        self.algo = algo 
        self.FLAGS = FLAGS
        self.exp_memory = ex_replay(memory_size = self.FLAGS.replay_size, batch_size = self.FLAGS.batch_size)
        
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
        if self.FLAGS.arch == 'REM_DQN':
            self.rco = tf.placeholder(tf.float32, [None])
            on_est_q, self.on_gd_idx, _ = self.algo.model(self.on_state, self.on_action, self.rco, 'online')
            _, _, tar_gd_action = self.algo.model(self.batch_ns, self.batch_action, self.rco, 'target')
            
        else:
            on_est_q, self.on_gd_idx, _ = self.algo.model(self.on_state, self.on_action, 'online')
            _, _, tar_gd_action = self.algo.model(self.batch_ns, self.batch_action, 'target')
            
        self.var_online = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'online')
        self.var_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'target')
        self.loss = self.algo.loss_fn(on_est_q, target_args = {'batch_rew': self.batch_reward,'batch_done':self.batch_done,
                                                               'gd_action_value':tar_gd_action})
        self.train_step = self.algo.dqn_optimizer(self.loss, optim_args={'lr':self.FLAGS.lr})
        
    def eps_greedy(self, obs, eps):
        greedy_action = self.sess.run(self.on_gd_idx, feed_dict = {self.on_state:obs})
        if rand.random() > eps:
            a_t = greedy_action[0]
        else:
            a_t = rand.randint(0, self.num_actions-1)
        return a_t
      
    def env_step(self, s_t, a_t, step_count): # execute action and save trainsition
        next_state, r_t, done, info = self.env.step(a_t)
        self.exp_memory.save_ex(s_t, a_t, np.sign(r_t), next_state, done, step_count = step_count)
        return next_state, r_t, done, info
        
    def perform_gd(self, step_count):
        # if memory collects enough data, start training (perform gradient descent with respect to online variable)
        if step_count > self.FLAGS.train_start and step_count%self.FLAGS.update_freq == 0:
            # load training data from memory with mini_batch size(=32)
            batch_s, batch_a, batch_r, batch_ns, batch_done = self.exp_memory.sample_ex(step_count)
                    
            # perform gradient descent with respect to online variable
            if self.FLAGS.arch == 'REM_DQN':
                random_coeff = make_coeff(self.FLAGS.num_heads)
                _, loss, loss_val = self.sess.run([self.train_step, self.tf_loss_summary, self.loss], 
                                                  feed_dict = {self.on_state:batch_s, self.on_action:batch_a, self.batch_ns:batch_ns, 
                                                               self.batch_reward:batch_r, self.batch_done:batch_done, self.rco:random_coeff})
            else:
                _, loss, loss_val = self.sess.run([self.train_step, self.tf_loss_summary, self.loss],
                                                  feed_dict = {self.on_state:batch_s, self.on_action:batch_a, self.batch_ns:batch_ns, 
                                                               self.batch_reward:batch_r, self.batch_done:batch_done})
            return loss, loss_val
        else:
            return None, None
    
    def train(self):
        # This function implements DQN with experience replay(Mnih et al. 2015)
        self.construct_graph()
        
        # initialize variables, assign target variables to online variables
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
        while step_count < self.FLAGS.max_frames:
            done = False
            step_start = step_count
            s_t = self.env.reset()
            if step_count == 0: # save the first frame in the first step, assign max_lives
                self.exp_memory.save_ex(s_t, None, None, None, None, step_count = step_count)
                max_lives = self.env.unwrapped.ale.lives()
                
            if self.env.unwrapped.ale.lives() == max_lives: # accumulate rewards until agent lost all lives
                rew_epi, loss_epi = 0, 0
                
            while done == False: # continue to step until an episode terminates
                # compute forward pass to find greedy action and select action with epsilon greedy strategy
                stacked_s = self.exp_memory.stack_frame(step_count = step_count, batch = False, b_idx = None)
                a_t = self.eps_greedy(stacked_s, eps)
                
                # execute action, observe reward and next state, save (s_t, a_t, r_t, s_t+1) to experience replay
                next_state, r_t, done, info = self.env_step(s_t, a_t, step_count)
                s_t = next_state
                
                # if memory collects enough data, perform a gradient descent step with respect to the online variable
                loss, loss_val = self.perform_gd(step_count)
                
                # Reset target_variable in every target_reset interval
                if step_count % self.FLAGS.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var_target, self.var_online)])
                
                # etc (decaying epsilon, acculates loss, rewards, increase step count and episode count)
                eps = linear_decay(step_count, start =self.FLAGS.eps, end = self.FLAGS.final_eps, frame = 1000000)
                if loss is not None:
                    tf_writer.add_summary(loss, step_count)
                    loss_epi += loss_val
                rew_epi += r_t
                step_count += 1
                
            if self.env.unwrapped.ale.lives() == 0:
                # save loss, reward per an episode, compute average reward on previous 100 number of episodes
                reward_his.append(rew_epi)
                avg_reward = np.mean(reward_his[-100:])
                mean_reward.append(avg_reward)
                avg_rew = self.sess.run(tf_rew_summary, feed_dict={avg_reward_log:avg_reward})
                tf_writer.add_summary(avg_rew, step_count)
                
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