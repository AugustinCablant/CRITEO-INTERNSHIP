from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from src.environment.tree import Tree, Node
import gym
from gym import spaces
import numpy as np

class StochasticEnvironment(gym.Env):
    """
    The Environment class represents a decision-making environment with a tree structure.
    The tree is generated dynamically and rewards are assigned to each node based on its level and value. 
    The environment allows the computation of rewards based on nodes and paths, 
    and finds the best strategy for decision-making.

    Attributes
    ----------
    env_rng : numpy.random.RandomState
        A random state generator used for sampling rewards.
    
    nb_layers : int 
        Number of layers for the Tree. 
        Example 3 

    nb_nodes_per_layer : list
        Number of nodes per layer for the Tree. 
        Example [1, 2, 5]

    mus : list
        Mean of the distribution for each layer. 
        Example [0, [1, 4], [[0.1, 0.3], [0.2, 0.1, 0.1]]]
    
    """ 
    def __init__(self, nb_nodes_per_layer, mus):
        self.env_rng = np.random.RandomState(2025)  # Use a fixed seed for reproducibility
        self.nb_layers = len(nb_nodes_per_layer) 
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.mus = mus
        self.sigma = 1

    def reset(self):
        self.tree = Tree()
        self.generate_tree()
        return [0]
    
    def _initialize_tree(self):
        """
        Initializes the tree by creating the root node.
        """
        self.root = None
        self.tree = Tree()
        self.root = self.tree.insert(self.root, ('root', self.mus[0])) 

    def generate_tree(self):
        self._initialize_tree()

        def build_layer(parents, mus_layer, layer_idx):
            current_layer_nodes = []

            if len(parents) != len(mus_layer):
                raise ValueError(f"Layer {layer_idx}: expected {len(parents)} parents, got {len(mus_layer)} mus sublists")
            
            for parent_idx, (parent_node, child_mus_list) in enumerate(zip(parents, mus_layer)):
                if not isinstance(child_mus_list, list):
                    raise ValueError(f"Layer {layer_idx}: mus[{layer_idx}][{parent_idx}] should be a list of floats")

                for child_idx, mu in enumerate(child_mus_list):
                    value = self.env_rng.normal(mu, self.sigma)
                    node_data = (f"Node_{layer_idx}_{parent_idx}_{child_idx}", value)
                    node, _ = self.tree.insert(parent_node, node_data)
                    current_layer_nodes.append(node)
            return current_layer_nodes

        if not isinstance(self.mus[0], (int, float)):
            raise ValueError("mus[0] must be a scalar for root node.")
        self.root.data = ("root", self.env_rng.normal(self.mus[0], self.sigma))

        parents = [self.root]

        for layer_idx in range(1, len(self.mus)):
            mus_layer = self.mus[layer_idx]

            if layer_idx == 1:
                current_layer_nodes = []
                for idx, mu in enumerate(mus_layer):
                    value = self.env_rng.normal(mu, self.sigma)
                    node_data = (f"Node_{layer_idx}_{idx}", value)
                    node, _ = self.tree.insert(self.root, node_data)
                    current_layer_nodes.append(node)
                parents = current_layer_nodes
            else:
                parents = build_layer(parents, mus_layer, layer_idx)

        return self.tree
    
    def get_action_set(self):
        return list(np.arange(len(self.tree.get_all_nodes())))
    
    def get_reward(self, action):
        """Obtain the reward given an action."""
        # The action is a path, with indexes corresponding to the nodes 
        node = self.tree.graph.get(action)
        if node:
            total_reward = 0
            path_node = node
            while path_node:
                total_reward += path_node.get_reward()
                path_node = path_node.parent
            return total_reward
        return 0
    
    def get_best_strategy_reward(self):
        all_rewards = [node.mean for node in self.tree.get_all_nodes()]
        return np.max(all_rewards)

    def get_reward_by_node(self, node):
        return node.get_reward()
    
    def step(self, action):
        reward = self.get_reward(action)
        done = True  
        return [0], reward, done, {}
    
    def render(self, mode='human', close=False):
        pass


            
        
        