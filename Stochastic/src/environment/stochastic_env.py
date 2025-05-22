from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from src.environment.tree import Tree, Node
import gym
from gym import spaces
import numpy as np

class StochasticEnvironment(gym.Env):
    def __init__(self, layers, min_children, max_children, p, reward_min, reward_max, var, seed=None):
        self.layers = layers
        self.min_children = min_children
        self.max_children = max_children
        self.p = p
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.var = var
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.tree = self.generate_tree()

    def reset(self):
        self.tree.step()

    def generate_tree(self):
        tree = Tree(self.seed)
        node_counter = 0  
        mu_root = self.rng.uniform(self.reward_min, self.reward_max)  # Root mean reward
        root_name = f"Node {node_counter}"
        root = tree.insert(parent_node=None, name=root_name, mean=mu_root, var=self.var)
        node_counter += 1

        def add_layer(parent_node, layer_idx):
            nonlocal node_counter
            if layer_idx >= self.layers:
                return

            num_children = self.rng.randint(self.min_children, self.max_children + 1)
            for i in range(num_children):
                if self.rng.rand() < self.p:  # Optimal reward
                    mean_reward = max(0, self.rng.uniform(self.reward_max - 0.5, self.reward_max + 0.5))
                else:  # Suboptimal reward
                    mean_reward = max(0, self.rng.uniform(self.reward_min - 0.1, self.reward_min + 0.1))

                node_name = f"Node {node_counter}"
                node_counter += 1

                # Adaptation ici : retour de insert peut être node ou (node, parent)
                inserted = tree.insert(parent_node, node_name, mean=mean_reward, var=self.var)
                if isinstance(inserted, tuple):
                    child_node = inserted[0]
                else:
                    child_node = inserted

                add_layer(child_node, layer_idx + 1)

        add_layer(root, 1)
        tree.step()
        return tree

    def get_action_set(self):
        return self.tree.get_all_leaves()
    
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
        self.tree.step()
    
    def render(self, mode='human', close=False):
        pass

""" 

class StochasticEnvironment(gym.Env):
    def __init__(self, layers, min_children, max_children, p, reward_min, reward_max, var, seed=None):
        self.layers = layers
        self.min_children = min_children
        self.max_children = max_children
        self.p = p
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.var = var
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.tree = self.generate_tree()

    def reset(self):
        self.tree.step()

    def generate_tree(self):
        tree = Tree(self.seed)
        node_counter = 0  
        mu_root = self.rng.uniform(self.reward_min, self.reward_max)  # Root mean reward
        root_name = f"Node {node_counter}"
        root = tree.insert(parent_node=None, name=root_name, mean=mu_root, var=self.var)
        node_counter += 1

        def add_layer(parent_node, layer_idx):
            nonlocal node_counter
            if layer_idx >= self.layers:
                return

            num_children = self.rng.randint(self.min_children, self.max_children + 1)
            for i in range(num_children):
                if self.rng.rand() < self.p:  # Optimal reward
                    mean_reward = max(0, self.rng.uniform(self.reward_max - 0.5, self.reward_max + 0.5))
                else:  # Suboptimal reward
                    mean_reward = max(0, self.rng.uniform(self.reward_min - 0.1, self.reward_min + 0.1))

                node_name = f"Node {node_counter}"
                node_counter += 1

                # Adaptation ici : retour de insert peut être node ou (node, parent)
                inserted = tree.insert(parent_node, node_name, mean=mean_reward, var=self.var)
                if isinstance(inserted, tuple):
                    child_node = inserted[0]
                else:
                    child_node = inserted

                add_layer(child_node, layer_idx + 1)

        add_layer(root, 1)
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
        return np.sum([node.value for node in path]), [node.value for node in path]
    
    def get_total_reward_per_leaf(self):
        action_set = self.get_action_set()
        total_reward_per_leaf = [self.get_reward(leaf) for leaf in action_set]
        self.total_reward_per_leaf = total_reward_per_leaf
        return total_reward_per_leaf
    
    def get_best_strategy_reward(self):
        best_arm_path =  self.tree.find_best_arm_path()
        return [node.name for node in best_arm_path] , np.sum([node.mean for node in best_arm_path])
    
    def step(self):
        self.tree.step()
    
    def render(self, mode='human', close=False):
        pass





class StochasticEnvironment(gym.Env):
    def __init__(self, mus, seed=None):
        self.mus = mus
        self.var = 2
        #self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.tree = self.generate_tree()
        
    def reset(self):
        self.tree = self.generate_tree()
    
    def generate_tree(self):  
        tree = Tree() ##
        mu_root = self.mus[0]
        root = tree.insert(parent_node=None, name="Targeting", mean=mu_root, var=0)
        var = 1

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
        return np.sum([node.value for node in path]), [node.value for node in path]
    
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

"""

