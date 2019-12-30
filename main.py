import numpy as np
import tensorflow as tf
import time
import gym
import model
import solver
from utils import *
from atari_env import *
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# Set configuration for the experiments
# Game environments
flags.DEFINE_string('game', 'PongNoFrameskip-v4', 'Atari environments') # 'Pong-v0'
flags.DEFINE_integer('skip_frame', 4, 'Number of frames skipped') 
flags.DEFINE_integer('update_freq', 4, 'Number of frames between each SGD')
# Model options
flags.DEFINE_string('arch', 'DQN', 'Nature DQN')
flags.DEFINE_integer('num_heads', 51, 'number of heads in the network')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
# Trainig method(offline, online), options
flags.DEFINE_bool('online', True, 'Training type, offline if False')
flags.DEFINE_float('eps', 1.0, 'epsilon start')
flags.DEFINE_float('final_eps', 0.1, 'final value of epsilon')
flags.DEFINE_float('eval_eps', 0.001, 'evaluation epsilon')
flags.DEFINE_string('eps_decay', 'linear', 'epsilon deacy')
flags.DEFINE_integer('train_start', 50000, 'train starts after this number of frame' )
flags.DEFINE_integer('target_reset', 10000, 'update frequency of target network')
flags.DEFINE_integer('replay_size', 1000000, 'experience replay size')
flags.DEFINE_integer('batch_size', 32, 'batch size for training')
# Solver options
flags.DEFINE_integer('max_frames', 6000000, 'maximum number of frames')
# Optimizer options
flags.DEFINE_string('opt', 'adam', 'Optimization method')
flags.DEFINE_float('lr', 0.00025, 'learning rate')
# Others
flags.DEFINE_string('exp', 'exp', 'log directory name')
flags.DEFINE_bool('verbose', True, 'print loss during trainig')
flags.DEFINE_bool('evaluate', False, 'evaluate trained agent ')
flags.DEFINE_integer('print_every', 10, 'print interval')
flags.DEFINE_integer('eval_every', 100, 'evaluation interval')
flags.DEFINE_integer('seed', 6550, 'seed number')

def main():
    sess = get_session()
    env = atari_env(FLAGS.seed, FLAGS.game)
    action_space = env.action_space.n
    
    logdir = './results/' + FLAGS.game + '_' + FLAGS.arch + '_seed' + str(FLAGS.seed) + '_' + FLAGS.exp + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    flags.DEFINE_string('logdir', logdir + '/', 'logdir')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    if FLAGS.arch == 'DQN':
        algo = model.DQN(num_actions = action_space, lr = FLAGS.lr, opt = FLAGS.opt, gamma = FLAGS.gamma, arch = FLAGS.arch)
        
    elif FLAGS.arch == 'C51':
        algo = model.C51(num_actions = action_space, lr = FLAGS.lr, num_heads = FLAGS.num_heads)
        
    elif FLAGS.arch == 'QR_DQN':
        algo = model.QR_DQN(num_actions = action_space, lr = FLAGS.lr, num_heads = FLAGS.num_heads) 
        
    elif FLAGS.arch == 'ENS_DQN':
        algo = model.ENS_DQN(num_actions = action_space, lr = FLAGS.lr, num_heads = FLAGS.num_heads)  
        
    elif FLAGS.arch == 'REM_DQN':
        algo = model.REM_DQN(num_actions = action_space, lr = FLAGS.lr, num_heads = FLAGS.num_heads)  
    
    if FLAGS.online:
        dqnsolver = solver.DQNsolver(env, sess, algo, FLAGS)
    else:
        dqnsolver = solver.offlineDQNsolver(env, sess, algo, FLAGS)
    dqnsolver.train()
    
if __name__ == "__main__":
    main()