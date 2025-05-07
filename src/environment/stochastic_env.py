from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from src.environment.tree import Tree, Node
import gym
from gym import spaces
import numpy as np

class StochasticEnvironment(gym.Env):
    def __init__(self, mus, seed=2025):
        self.mus = mus
        self.var = 2
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.generate_tree()
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed)
        tree = self.generate_tree()
        return tree
    
    def generate_tree(self):
        tree = Tree(seed=self.seed)
        var = 2
        root = tree.insert(parent_node=None, name="Utilisateur", mean=0, var=0)

        # First Layer
        mu_layer1 = self.mus[0]
        vehicule, _ = tree.insert(parent_node=root, name="vehicule", mean=mu_layer1[0], var=var)
        mob_verte, _ = tree.insert(parent_node=root, name="Mobilités Vertes", mean=mu_layer1[1], var=var)

        # Second Layer
        mu_layer2 = self.mus[1]
        mu_layer2_1 = mu_layer2[0]
        taxi, _ = tree.insert(parent_node=vehicule, name="taxi", mean=mu_layer2_1[0], var=var)
        uber, _ = tree.insert(parent_node=vehicule, name="uber", mean=mu_layer2_1[1], var=var)
        bus, _ = tree.insert(parent_node=vehicule, name="bus", mean=mu_layer2_1[2], var=var)
        tram, _ = tree.insert(parent_node=vehicule, name="tram", mean=mu_layer2_1[3], var=var)
        covoiturage, _ = tree.insert(parent_node=vehicule, name="covoiturage", mean=mu_layer2_1[4], var=var)
        avion, _ = tree.insert(parent_node=vehicule, name="avion", mean=mu_layer2_1[5], var=var)
        train, _ = tree.insert(parent_node=vehicule, name="train", mean=mu_layer2_1[6], var=var)
        rer, _ = tree.insert(parent_node=vehicule, name="RER", mean=mu_layer2_1[7], var=var)
        metro, _ = tree.insert(parent_node=vehicule, name="métro", mean=mu_layer2_1[8], var=var)

        mu_layer2_2 = mu_layer2[1]
        velo, _ = tree.insert(parent_node=mob_verte, name="vélo", mean=mu_layer2_2[0], var=var)
        marche, _ = tree.insert(parent_node=mob_verte, name="marche", mean=mu_layer2_2[1], var=var)
        cap, _ = tree.insert(parent_node=mob_verte, name="course à pied", mean=mu_layer2_2[2], var=var)
        
        self.tree = tree
        return tree
    
    
    def get_action_set(self):
        return self.tree.get_all_leaves()
    
    def sample_randomly(self):
        action_set = self.get_action_set()
        random_index = np.random.choice(len(action_set))
        leaf_chosen = action_set[random_index]
        return leaf_chosen

    def get_reward(self, leaf):
        return self.tree.get_reward_leaf(leaf)
    
    def get_total_reward_per_leaf(self):
        action_set = self.get_action_set()
        total_reward_per_leaf = [self.get_reward(leaf) for leaf in action_set]
        self.total_reward_per_leaf = total_reward_per_leaf
        return total_reward_per_leaf
    
    def get_best_strategy_reward(self):
        best_arm_path =  self.tree.find_best_arm_path()
        return [node.name for node in best_arm_path] ,np.sum([node.value for node in best_arm_path])
    
    def get_gap(self, leaf):
        total_reward_per_leaf = self.get_total_reward_per_leaf()
        reward_leaf = self.get_reward(leaf)
        return max(total_reward_per_leaf) - reward_leaf
    
    def get_means(self):
        return list(np.array(self.mus[0][0]) + np.array(self.mus[1][0])) + list(np.array(self.mus[0][1]) + np.array(self.mus[1][1]))

    def step(self, leaf):
        reward = self.get_reward_leaf(leaf)
        done = True  
        return [0], reward, done, {}
    
    def render(self, mode='human', close=False):
        pass