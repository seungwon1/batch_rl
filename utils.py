import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image
from atari_env import set_atari_env

def atari_env(seed_number, env, life_terminate = True):
    os.environ['PYTHONHASHSEED']=str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.set_random_seed(seed_number)
    env = gym.make(env)
    env.seed(seed_number)
    env = set_atari_env(env, life_terminate)
    return env

def get_session(): # use with get_session() as sess: or sess = get_session()
    from tensorflow.compat.v1 import InteractiveSession
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    #session = InteractiveSession(config=config)
    return session
    
def preprocess(images): # input : raw image of shape (N, 210, 160, 3) from gym atari env
    images = np.reshape(np.array(images), [1, 210, 160, 3]).astype(np.float32)
    rgb2y = 0.299*images[...,0] + 0.587*images[...,1] + 0.114*images[...,2] # compute luminance from rgb
    resized_input = cv2.resize(rgb2y.reshape(210, 160), (84, 110)) # resize with bilinear interpolation (default) cv2.resize(rgb2y.reshape(210, 160), (84, 84))  # interpolation=cv2.INTER_LINEAR
    resized_input = resized_input[18:102, :].astype(np.uint8) # convert to 0-255 scale
    return resized_input # output of shape (84,84)

def linear_decay(step_count, start=1, end=0.1, frame=1000000):
    eps = start-(start-end)/(frame-1)*(step_count+1)
    if step_count > (frame-2):
        eps = end
    return eps

class ex_replay(object): # experience replay
    
    def __init__(self, memory_size, batch_size = 32):
        self.memory_size = memory_size
        self.memory_frame = np.empty((memory_size, 84, 84), dtype=np.uint8)
        self.memory_a_r = np.empty((memory_size, 3))
        self.batch_size = batch_size
        
    def save_ex(self, s_t, a_t, r_t, next_s_t, done, step_count): 
        """
        Input: s_t and next_s - each of shape (210, 160, 3) (raw observation from env), a_t and r_t
        Save unit frame of shape 84*84 instead of saving stacked frames
        """
        if step_count == 0:
            self.memory_frame[step_count%self.memory_size, :, :] = s_t.reshape(84,84) #preprocess(s_t)
        else:
            self.memory_frame[(step_count+1)%self.memory_size, :, :] = next_s_t.reshape(84,84) #preprocess(next_s_t)
            self.memory_a_r[step_count%self.memory_size, 0], self.memory_a_r[step_count%self.memory_size, 1] = a_t, r_t
            self.memory_a_r[step_count%self.memory_size, 2] = done
    
    def stack_frame(self, b_idx, step_count = None, batch = True):
        """
        Stack 4 most recent frames, return 0 array if there is no frame in the buffer or previous frame is not in the same game(episode)
        Input: index of batch sample of shape (N, ), Ouput: of shape (N, 84, 84, 4)
        if batch = False, stack for a single data frame to compute forward pass of NN for selecting greedy action during exploration
        """ 
        if batch:
            out_frame = np.empty((self.batch_size, 84, 84, 4), dtype = np.uint8)
            for idx_k, j in enumerate(b_idx):
                for i in range(4):
                    out_frame[idx_k,:,:,i] = self.memory_frame[(j-(3-i)+self.memory_size)%self.memory_size]
                    if self.memory_a_r[(j-(3-i)+self.memory_size)%self.memory_size, 2] == 1 and i < 3: # return zero array if done 
                        out_frame[idx_k,:,:,:i+1] = np.zeros_like(out_frame[idx_k,:,:,:i+1])
                               
                    if step_count < self.memory_size and j - (3-i) < 0: # return zero array if idx is negative value
                        out_frame[idx_k, :, :, i] = np.zeros_like(out_frame[idx_k, :, :, i])  
                        
        else:
            out_frame = np.empty((1, 84, 84, 4), dtype = np.uint8)
            for i in range(4):
                out_frame[..., i] = self.memory_frame[(step_count-(3-i)+self.memory_size)%self.memory_size]
                if self.memory_a_r[(step_count-(3-i)+self.memory_size)%self.memory_size, 2] == 1 and i < 3: # return zero array if done 
                    out_frame[..., :i+1] = np.zeros_like(out_frame[..., :i+1])
                    
                if step_count < self.memory_size and step_count - (3-i) < 0: # return zero array if idx is negative value
                    out_frame[..., i] = np.zeros_like(out_frame[..., i])
        return out_frame
   
    def sample_ex(self, step_count, training_type = 'online'):
        "Sample experience from replay. Output exprience with batch_size"
        if training_type == 'online':
            if step_count < self.memory_size:
                b_idx = np.random.choice(step_count+1, self.batch_size)
            else:
                b_idx = np.random.choice(self.memory_size, self.batch_size)
                
        elif training_type == 'offline':
            b_idx = np.random.choice(self.memory_size, self.batch_size)
         
        s_t, next_s_t = self.stack_frame(b_idx, step_count = step_count), self.stack_frame(b_idx+1, step_count = step_count) # each of shape (N,84,84,4), which is input of NN
        a_t, r_t, done_t = self.memory_a_r[b_idx, 0], self.memory_a_r[b_idx, 1], self.memory_a_r[b_idx, 2]
        return s_t, a_t, r_t, next_s_t, done_t
