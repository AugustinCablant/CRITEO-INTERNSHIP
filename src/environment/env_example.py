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

class StochasticEnvironmentExample(gym.Env):
    def __init__(self, mus, var, patho=False, seed=None):
        self.mus = mus
        self.var = var
        #self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.patho = patho
        self.tree = None
        self.reset()
        
    def reset(self):
        if self.patho:
            self.tree = self.generate_tree_patho()
        else:
            self.tree = self.generate_tree()
    
    def generate_tree(self):  
        tree = Tree() ##
        mu_root = self.mus[0]
        root = tree.insert(parent_node=None, name="Targeting", mean=mu_root, var=0)
        var = self.var

        # First Layer
        mus_layer1 = self.mus[1]
        families, _ = tree.insert(parent_node=root, name="families", mean=mus_layer1[0], var=var)
        professionals, _ = tree.insert(parent_node=root, name="professionals", mean=mus_layer1[1], var=var)

        # Second Layer
        mus_layer2 = self.mus[2]
        mus_layer21 = mus_layer2[0]
        adults, _ = tree.insert(parent_node=families, name="adults", mean=mus_layer21[0], var=var)
        youngs, _ = tree.insert(parent_node=families, name="youngs", mean=mus_layer21[1], var=var)

        mus_layer22 = mus_layer2[1]
        start_up, _ = tree.insert(parent_node=professionals, name="start-up", mean=mus_layer22[0], var=var)
        companies, _ = tree.insert(parent_node=professionals, name="companies", mean=mus_layer22[1], var=var)

        # Third Layer
        mus_layer3 = self.mus[3]
        mus_layer31 = mus_layer3[0]
        mus_layer311 = mus_layer31[0]
        radio, _ = tree.insert(parent_node=adults, name="radio", mean=mus_layer311[0], var=var)
        tv, _ = tree.insert(parent_node=adults, name="tv", mean=mus_layer311[1], var=var)
        magazines, _ = tree.insert(parent_node=adults, name="magazines", mean=mus_layer311[2], var=var)
        tv, _ = tree.insert(parent_node=adults, name="supermarket", mean=mus_layer311[3], var=var)

        mus_layer312 = mus_layer31[1]
        youtube, _ = tree.insert(parent_node=youngs, name="youtube", mean=mus_layer312[0], var=var)
        social_networks, _ = tree.insert(parent_node=youngs, name="social networks", mean=mus_layer312[1], var=var)
        
        mus_layer32 = mus_layer3[1]
        mus_layer321 = mus_layer32[0]
        webinaire, _ = tree.insert(parent_node=start_up, name="webinaire", mean=mus_layer321[0], var=var)
        events, _ = tree.insert(parent_node=start_up, name="events", mean=mus_layer321[1], var=var)

        mus_layer322 = mus_layer32[1]
        linkedin, _ = tree.insert(parent_node=companies, name="linkedin", mean=mus_layer322[0], var=var)
        email, _ = tree.insert(parent_node=companies, name="email", mean=mus_layer322[1], var=var)


        tree.step()
        return tree
    
    def generate_tree_patho(self):
        tree = Tree()
        mu_root = self.mus[0]
        root = tree.insert(parent_node=None, name="Targeting", mean=mu_root, var=0)

        # First Layer
        mus_layer1 = self.mus[1]
        non_optimal_node, _ = tree.insert(parent_node=root, name="Non Optimal Node", mean=mus_layer1[0], var=self.var)
        optimal_node, _ = tree.insert(parent_node=root, name="Optimal Node", mean=mus_layer1[1], var=self.var)

        # Second Layer
        mus_layer2 = self.mus[2]
        mus_layer21 = mus_layer2[0]
        for i, mu in enumerate(mus_layer21):
            non_optimal_child, _ = tree.insert(parent_node=non_optimal_node, name=f"Non Optimal Child n°{i+1}", mean=mu, var=self.var)
        mus_layer22 = mus_layer2[1]
        optimal_child, _ = tree.insert(parent_node=optimal_node, name=f"optimal_child", mean=mus_layer22[0], var=self.var)
        tree.step()

        return tree
    
    def get_action_set(self):
        return self.tree.get_all_leaves() #
    
    def sample_randomly(self):
        action_set = self.get_action_set()
        random_index = np.random.choice(len(action_set))
        leaf_chosen = action_set[random_index]
        return leaf_chosen

    def get_reward_mu(self, index):
        reward_leaves = self.tree.get_mu_leaves()
        _, _, reward = reward_leaves[index]
        return reward
    
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
        if self.patho:
            self.tree = self.generate_tree_patho()
        else:
            self.tree = self.generate_tree()
    
    def render(self, mode='human', close=False):
        pass



