import numpy as np
import tensorflow as tf
import gym
import model
import solver
from utils import *
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
 
# Set configuration for the experiments
# Game environments
flags.DEFINE_string('game', 'PongNoFrameskip-v4', 'Atari environments') # 'Pong-v0'

# Model options
flags.DEFINE_string('arch', 'DQN', 'Nature DQN')
flags.DEFINE_integer('num_heads', 51, 'number of heads in the network')
flags.DEFINE_float('gamma', 0.99, 'discount factor')

# Trainig method(offline, online), options
flags.DEFINE_string('setting', 'offline', 'Training method')
flags.DEFINE_bool('online', True, 'Training type, offline if False')
flags.DEFINE_float('eps', 0.99, 'epsilon start')
flags.DEFINE_float('final_eps', 0.1, 'final value of epsilon')
flags.DEFINE_string('eps_decay', 'linear', 'epsilon deacy')
flags.DEFINE_integer('train_start', 50000, 'train starts after this number of frame' )
flags.DEFINE_integer('target_reset', 10000, 'update frequency of target network')
flags.DEFINE_integer('replay_size', 1000000, 'experience replay size')
flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_string('loss_ft', 'huber', 'Loss function')

# Solver options
flags.DEFINE_integer('max_episodes', 1000, 'maximum number of episodes')
flags.DEFINE_integer('max_frames', 50000000, 'maximum number of frames')

# Optimizer options
flags.DEFINE_string('opt', 'adam', 'Optimization method') # rmsprop
flags.DEFINE_float('lr', 0.00025, 'learning rate')
flags.DEFINE_bool('clip', True, 'gradient clipping')

# Others
flags.DEFINE_bool('verbose', True, 'print loss during trainig')
flags.DEFINE_integer('print_every', 10, 'print interval')
flags.DEFINE_integer('eval_every', 100, 'evaluation interval')
flags.DEFINE_integer('seed', 6550, 'seed number')

def main():
    sess = get_session()
    env = set_seed(FLAGS.seed, FLAGS.game)
    action_space = env.action_space.n
    
    algo = model.DQN(num_actions = action_space, num_atoms = FLAGS.num_heads, lr = FLAGS.lr, 
                     opt = FLAGS.opt, clipping = FLAGS.clip, arch = FLAGS.arch)
    
    state, action, q_val, est_q, greedy_idx, greedy_action = algo.model('online') # online network
    pkg1 = (state, action, q_val, est_q, greedy_idx, greedy_action)
    
    state_target, max_q_target = algo.model('target') # target network
    pkg2 = (state_target, max_q_target)
    
    mean_loss = algo.dqn_loss(q_val, est_q, loss_type = FLAGS.loss_ft)  # logits (true): q_val (fixed target), pred: est_q
    train_step = algo.dqn_optimizer(mean_loss)
        
    var_online = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'online')
    var_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'target')
    sess.run(tf.global_variables_initializer())    
    sess.run( [tf.assign(t, o) for t, o in zip(var_target, var_online)])

    dqnsolver = solver.dqn_online_solver(env, train_step, mean_loss, action_space, var_online, var_target, sess, 
                            FLAGS.batch_size, FLAGS.train_start, FLAGS.target_reset, FLAGS.max_frames, FLAGS.eps, 
                            FLAGS.eps_decay, FLAGS.gamma, FLAGS.replay_size, pkg1, pkg2,
                            FLAGS.arch, FLAGS.print_every, FLAGS.eval_every, FLAGS.verbose)
    
    loss_his, reward_his, reward100, eval_his, best_param_list, final_var = dqnsolver.train()
    
if __name__ == "__main__":
    main()   