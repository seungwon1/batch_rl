import numpy as np
import tensorflow as tf
import gym
import model
import solver
from utils import *
from atari_wrappers import *
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
 
# Set configuration for the experiments
# Game environments
flags.DEFINE_string('game', 'PongNoFrameskip-v4', 'Atari environments') # 'Pong-v0'
flags.DEFINE_integer('skip_frame', 4, 'Number of frames skipped') 
flags.DEFINE_integer('update_freq', 4, 'Number of frames between each SGD') 
flags.DEFINE_integer('no_op', 30, 'Number of random actions executed before the agent starts an episode') 

# Model options
flags.DEFINE_string('arch', 'DQN', 'Nature DQN')
flags.DEFINE_integer('num_heads', 51, 'number of heads in the network')
flags.DEFINE_float('gamma', 0.99, 'discount factor')

# Trainig method(offline, online), options
flags.DEFINE_string('setting', 'offline', 'Training method')
flags.DEFINE_bool('online', True, 'Training type, offline if False')
flags.DEFINE_float('eps', 1, 'epsilon start')
flags.DEFINE_float('final_eps', 0.1, 'final value of epsilon')
flags.DEFINE_string('eps_decay', 'linear', 'epsilon deacy')
flags.DEFINE_integer('train_start', 50000, 'train starts after this number of frame' )
flags.DEFINE_integer('target_reset', 10000, 'update frequency of target network')
flags.DEFINE_integer('replay_size', 1000000, 'experience replay size')
flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_string('loss_ft', 'huber', 'Loss function')

# Solver options
flags.DEFINE_integer('max_frames', 6000000, 'maximum number of frames') # 50000000

# Optimizer options
flags.DEFINE_string('opt', 'rmsprop', 'Optimization method') # adam
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_bool('clip', True, 'gradient clipping')

# Others
flags.DEFINE_bool('verbose', True, 'print loss during trainig')
flags.DEFINE_bool('reload', False, 'load previous session ')
flags.DEFINE_bool('evaluate', False, 'evaluate trained agent ')
flags.DEFINE_integer('print_every', 10, 'print interval')
flags.DEFINE_integer('eval_every', 100, 'evaluation interval')
flags.DEFINE_integer('seed', 6550, 'seed number')
flags.DEFINE_bool('fast_test', True, 'test mode for faster convergence') # set False and FLAGS.clip as False to make Nature DQN ENV

def main():
    sess = get_session()
    env = set_seed(FLAGS.seed, FLAGS.game)
    action_space = env.action_space.n
    
    if FLAGS.arch == 'DQN':
        algo = model.DQN(num_actions = action_space, lr = FLAGS.lr, 
                     opt = FLAGS.opt, clipping = FLAGS.clip, arch = FLAGS.arch)
        
        state, action, est_q, greedy_idx, greedy_action = algo.model('online') # online network
        pkg1 = (state, action, est_q, greedy_idx, greedy_action)
    
        batch_state, batch_reward, batch_done, max_q_target = algo.model('target') # target network
        pkg2 = (batch_state, batch_reward, batch_done, max_q_target, None)
        
        var_online = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'online')
        var_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'target')
        
        mean_loss = algo.dqn_loss(tf.stop_gradient(max_q_target), est_q, loss_type = FLAGS.loss_ft)  # logits (true):q_val(fixed target), pred: est_q
        train_step, lr = algo.dqn_optimizer(mean_loss, var_online)
        
    elif FLAGS.arch == 'C51':
        algo = model.C51(num_actions = action_space) #, num_heads = FLAGS.num_heads, lr = FLAGS.lr, opt = FLAGS.opt, clipping = FLAGS.clip, arch = FLAGS.arch)
        state, action, est_q_online, greedy_idx, raw_est_q = algo.model('online') # online network
        pkg1 = (state, action, est_q_online, greedy_idx, raw_est_q)
    
        batch_ns, batch_rew, batch_done, update_support, est_q_target = algo.model('target') # target network
        pkg2 = (batch_ns, batch_rew, batch_done, update_support, est_q_target)
                
        var_online = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'online')
        var_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'target')

        project_prob = algo.categorical_algorithm(q_online=est_q_online, q_target=est_q_target,upt_support=update_support)
        mean_loss = algo.c51_loss(project_prob, raw_est_q)
        train_step, lr = algo.dqn_optimizer(mean_loss, var_online)

    sess.run(tf.global_variables_initializer())    
    sess.run([tf.assign(t, o) for t, o in zip(var_target, var_online)])
    
    dqnsolver = solver.dqn_online_solver(env, train_step, lr, mean_loss, action_space, var_online, var_target, 
                                         sess, pkg1, pkg2, FLAGS)
    loss_his, reward_his, mean_reward, eval_his = dqnsolver.train()
    
if __name__ == "__main__":
    main()   