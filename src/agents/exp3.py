import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from itertools import combinations
import numpy as np
import time
from tqdm import tqdm
import psutil
from src.utils.save_results import save_result

class Exp3:
    def __init__(self, action_set):
        self.action_set = action_set
        self.K = len(self.action_set)
        self.EPS = 1e-8
        self.reset()     

    def reset(self, seed=2025):
        self.rng = np.random.RandomState(seed) 
        self.total_reward = 0
        self.t = 0  
        self.count_actions = np.zeros(self.K)
        self.count_rewards = np.zeros(self.K)
        initial_dist = np.full(self.K, 1.0 / self.K)
        self.weights = initial_dist
        self.lr = 1
        self.proba_t = np.zeros(self.K)

    def vector_proba(self, y):
        stable_exp_y = np.exp(y - np.max(y))
        proba_vector = stable_exp_y / np.sum(stable_exp_y)
        return proba_vector
    
    def get_action(self, round):
        self.define_lr(round)
        y = self.weights * self.lr
        self.proba_t = self.vector_proba(y)
        idx_list = range(self.K)
        action = self.rng.choice(idx_list, p=self.proba_t)  # index 
        return action
    
    def define_lr(self, round):
        self.lr = 1/np.sqrt(round+1)

    def receive_reward(self, reward, action_index):
        # We can now update the scores 
        self.weights[action_index] += (reward) / (self.proba_t[action_index] + self.EPS)

        self.total_reward += reward
        self.count_rewards[action_index] += reward

    def nested(self):
        return False
    
    def name(self):
        return 'Exp3'