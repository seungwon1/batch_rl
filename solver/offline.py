import numpy as np
import tensorflow as tf
import math
import gym
from utils import *

class dqn_offline_solver(object):
    
    def __init__(self, env, train_step, loss, num_actions, variable, var2, sess, mini_batch, train_start, target_reset, max_frames, epsilon, epsilon_decay, gamma, memory_capa, pkg1, pkg2, arch, print_every = 100, eval_every = 50, verbose = True):
        
        self.env = env
        self.train_step = train_step
        self.mean_loss = loss
        self.var = variable
        self.var2 = var2
        self.sess = sess
        self.num_actions = num_actions

        self.mini_batch = mini_batch
        self.train_start = train_start
        self.target_reset = target_reset
        self.max_frames = max_frames
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.memory_capa = memory_capa
        self.arch = arch
        
        self.print_every = print_every
        self.eval_every =eval_every
        self.verbose = verbose
        self.state, self.action, self.q_val, self_est_q, self.batch_size, self.gd_idx, self.gd_action  = pkg1
        self.state_target, self.max_q_target = pkg2
        
    def train(self):
        step_count, episode_count = 0, 0
        exp_memory = ex_replay(memory_size = self.memory_capa) # initialize experience replay
        loss_his, reward_his, eval_his, reward_100, best_param_list = [], [], [], [], []
        prev_avg_reward = 0
        final_param = None
        eps = self.epsilon
        time1 = time.time()
        saver = tf.train.Saver()

        while step_count < self.max_frames:
            rew_epi, loss_epi = 0, 0   
            done = False
            step_start = step_count
            s_t = self.env.reset()   
            if step_count == 0: # only save first frame
                exp_memory.save_ex(s_t, None, None, None, None, step_count = step_count)
                
            while done == False: # continue to step until an episode terminates
                if step_count % 4 == 0: # select action in every kth frame
                    # compute forward pass to find greedy action and select action with epsilon greedy strategy
                    stacked_s = exp_memory.stack_frame(b_idx = None, step_count = step_count, batch = False)
                    greedy_action = self.sess.run([self.gd_idx], feed_dict = {self.state:stacked_s, self.action:np.ones((1,)), self.batch_size:1})
                    action_probs = (1/self.num_actions)*eps*np.ones((self.num_actions,))
                    action_probs[tuple(greedy_action)] += 1-eps
                    a_t = np.random.choice(self.num_actions, 1, p = action_probs)[0]
                    
                next_state, r_t, done, info =  self.env.step(a_t) # step ahead

                # load training data from memory with mini_batch size(=32)
                batch_s, batch_a, batch_r, batch_ns, batch_done = exp_memory.sample_ex(step_count, training_type = 'offline')
                    
                # use fixed target variable to calculate target max_Q
                max_q_hat = self.sess.run(self.max_q_target, feed_dict = {self.state_target:batch_ns})
                target = batch_r + self.gamma*max_q_hat # target_q_val, of shape (minibatch, 1)
                target[batch_done == 1] = batch_r[batch_done == 1] # assign reward only if next state is terminal

                # perform gradient descent to online variable
                _, loss = self.sess.run([self.train_step, self.mean_loss], feed_dict = {self.state:batch_s, self.action:batch_a, self.batch_size:self.mini_batch, self.q_val:target})
                loss_epi += loss

                # linearly decaying epsilon for every step
                eps = linear_decay(step_count)
                
                # Reset target_variables in every interval(target_reset)
                if step_count % self.target_reset == 0:
                    self.sess.run( [tf.assign(t, o) for t, o in zip(self.var2, self.var)])

                # increase step count, accumulates rewards
                step_count += 1
                rew_epi += r_t

            # save loss, reward per an episode, compute average reward on previous 100 number of episodes
            loss_his.append(loss_epi)
            reward_his.append(rew_epi)
            prev_avg_reward = np.mean(reward_his[-100:])
            reward_100.append(prev_avg_reward)
            
            # print progress if verbose is True
            if self.verbose:
                if episode_count == 0:
                    print('\nStart training '+ str(self.arch))
                if episode_count % self.print_every == 0:
                    print('\nEpisode {0}: reward {1:2g}, loss {2:2g}, epsilon {3:2g}, steps {4}, total steps {5}'\
                          .format(episode_count+1 ,rew_epi,loss_epi, eps, step_count - step_start, step_count))
                    
                    time2 = time.time()
                    print('time (minutes) :', int((time2-time1)/60))
                    plt.plot(reward_his, label = 'reward')
                    plt.title('Iteration frame: '+ str(step_count))
                    plt.legend()
                    plt.savefig('./results/it_frame_reward'+str(step_count))
                    plt.clf()
                    
                    plt.plot(loss_his, label = 'loss')
                    plt.title('Iteration frame: '+ str(step_count))
                    plt.legend()
                    plt.savefig('./results/it_frame_loss'+str(step_count))
                    plt.clf()
               
                if episode_count % 50 == 0:
                    saver.save(self.sess, "./tmp/model", global_step=step_count)

            # increase episode_count
            episode_count += 1
            
        return loss_his, reward_his, reward_100, eval_his, best_param_list, final_param