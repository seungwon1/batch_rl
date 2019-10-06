import numpy as np
import tensorflow as tf
import gym
import cv2
import random
from collections import deque
# reference: https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/atari_wrappers.py

class random_start_env(gym.Wrapper):
    def __init__(self, env, random_max = 30):
        super().__init__(env)
        self.random_max = random_max
        
    def reset(self):
        self.env.reset()
        num_rand = np.random.randint(1,self.random_max + 1, size = 1)[0]
        for _ in range(num_rand):
            obs, reward, done, info = self.env.step(0)
        return obs
        
class max_skip_env(gym.Wrapper):
    def __init__(self, env, frame_skip = 4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.obs_list = deque(maxlen=2)
        
    def step(self, action):
        rew = 0.0
        for t in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            rew += reward
            self.obs_list.append(obs)
            if done:
                break   
        obs = np.max(np.stack(self.obs_list), axis=0)
        return obs, rew, done, info     
    
    def reset(self):
        self.obs_list.clear()
        obs = self.env.reset()
        self.obs_list.append(obs)
        return obs
    
class preprocess_clip_env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), np.sign(reward), done, info     
    
    def reset(self):
        return self.preprocess(self.env.reset())    
    
    def preprocess(self, obs):
        images = np.reshape(np.array(obs), [1, 210, 160, 3]).astype(np.float32)
        rgb2y = 0.299*images[...,0] + 0.587*images[...,1] + 0.114*images[...,2] # compute luminance from rgb
        resized_input = cv2.resize(rgb2y.reshape(210, 160), (84, 84)) # resize with bilinear interpolation (default)
        resized_input = resized_input.astype(np.uint8) # convert to 0-255 scale
        return resized_input # output of shape (84,84)    

class life_termination(gym.Wrapper):
    def __init__(self, env):
        "Terminate episode when the agent lost life"
        "Reset on true game over"
        super().__init__(env)
        self.lives = 0
        self.true_game_over = True
        self.real_reset = False
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.true_game_over = done
        
        lives = self.env.unwrapped.ale.lives()
        if lives > 0 and lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self):
        if self.true_game_over:
            obs = self.env.reset()
            self.real_reset = True
        else:
            obs, _, _, _ = self.env.step(0)
            self.real_reset = False
            
        self.lives = self.env.unwrapped.ale.lives()
        return obs
        
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs    

def set_atari_env(env, life_terminate = False):
    if life_terminate:
        env = life_termination(env)
    env = random_start_env(env)
    env = max_skip_env(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = preprocess_clip_env(env)
    return env
