import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from src.environment.tree import Tree, Node
import gym
from gym import spaces
import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class BestOfBothWorldEnvironment(gym.Env):
    def __init__(self):
        self.rng = np.random.default_rng()
        self.tree = None
        self.reset()
        
    def reset(self):
        self.tree = self.generate_tree()
    
    def generate_tree(self):  
        tree = Tree(uniform=True) ##
        root = tree.insert(parent_node=None, name="Targeting")

        # First Layer
        families, _ = tree.insert(parent_node=root, name="families")
        professionals, _ = tree.insert(parent_node=root, name="professionals", best=True)

        # Second Layer
        adults, _ = tree.insert(parent_node=families, name="adults")
        youngs, _ = tree.insert(parent_node=families, name="youngs")

        start_up, _ = tree.insert(parent_node=professionals, name="start-up", best=True)
        companies, _ = tree.insert(parent_node=professionals, name="companies")

        # Third Layer
        radio, _ = tree.insert(parent_node=adults, name="radio")
        tv, _ = tree.insert(parent_node=adults, name="tv")
        magazines, _ = tree.insert(parent_node=adults, name="magazines")
        tv, _ = tree.insert(parent_node=adults, name="supermarket")

        youtube, _ = tree.insert(parent_node=youngs, name="youtube")
        social_networks, _ = tree.insert(parent_node=youngs, name="social networks")
        
        webinaire, _ = tree.insert(parent_node=start_up, name="webinaire")
        events, _ = tree.insert(parent_node=start_up, name="events", best=True)

        linkedin, _ = tree.insert(parent_node=companies, name="linkedin")
        email, _ = tree.insert(parent_node=companies, name="email")
        tree.step()
        return tree
    
    
    def get_action_set(self):
        return self.tree.get_all_leaves() 
    
    def sample_randomly(self):
        action_set = self.get_action_set()
        random_index = np.random.choice(len(action_set))
        leaf_chosen = action_set[random_index]
        return leaf_chosen
    
    def get_reward(self, index):
        reward_leaves = self.tree.get_reward_leaves()
        _, _, reward = reward_leaves[index]
        return reward
    
    def get_reward_by_path(self, path):
        return np.sum([node.value for node in path]), [node.value for node in path], np.sum([node.mean for node in path])
    
    def get_total_reward_per_leaf(self):
        action_set = self.get_action_set()
        total_reward_per_leaf = [self.get_reward(leaf) for leaf in action_set]
        self.total_reward_per_leaf = total_reward_per_leaf
        return total_reward_per_leaf
    
    def get_best_strategy_reward(self):
        best_arm_path =  self.tree.find_best_arm_path()
        return [node.name for node in best_arm_path] , np.sum([node.mean for node in best_arm_path])
    
    def step(self):
        self.tree = self.generate_tree()
    
    def render(self, mode='human', close=False):
        pass