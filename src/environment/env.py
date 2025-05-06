import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from src.environment.tree import Tree, Node


import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    """
    The Environment class represents a decision-making environment with a tree structure.
    The tree is generated dynamically and rewards are assigned to each node based on its level and value. 
    The environment allows the computation of rewards based on nodes and paths, 
    and finds the best strategy for decision-making.

    Attributes
    ----------
    sampling_rng : numpy.random.RandomState
        A random state generator used for sampling rewards.
    env_rng : numpy.random.RandomState
        A fixed random state generator for the environment.
    nb_leaves_per_class : int
        The number of leaves (children) per node at each level.
    nb_levels : int
        The maximum number of levels in the tree.
    root : Node
        The root node of the tree.
    tree : Tree
        The tree structure containing all the nodes.
    best_strategy_path : list
        The best path (indices of the children) to follow for maximum reward.
    best_strategy_nodes_path : list
        The nodes corresponding to the best strategy path.
    node_distribution : str
        The distribution strategy for the number of children per node ('random', 'balanced', 'imbalanced')
    """

    def __init__(self, rd, nb_leaves_per_class, nb_levels, node_distribution):
        """
        Initializes the Environment object with the given settings.

        Parameters
        ----------
        rd : int
            Random state seed for reward sampling.
        nb_leaves_per_class : int
            Number of leaves (children) per node.
        nb_levels : int
            The number of levels in the tree.
        node_distribution : str
            Determines how the number of children are distributed ('random', 'balanced', 'imbalanced').
        """
        self.sampling_rng = np.random.RandomState(rd)
        self.env_rng = np.random.RandomState(2025)

        self.nb_leaves_per_class = nb_leaves_per_class
        self.nb_levels = nb_levels
        self.node_distribution = node_distribution  # 'random', 'balanced', 'imbalanced'

        self._initialize_tree()

    def _initialize_tree(self):
        """
        Initializes the tree by creating the root node.
        """
        self.root = None
        self.tree = Tree()
        self.root = self.tree.insert(self.root, ('root', 0))

    def set(self):
        """
        Sets up the environment by creating the tree iteratively and computing the best strategy.
        The best strategy is the path with the maximum reward.

        Updates the following attributes:
            - best_strategy_path : The path of indices representing the best strategy.
            - best_strategy_nodes_path : The list of nodes corresponding to the best strategy.
        """
        self.iterative_graph_create()
        _, self.best_strategy_path = self.tree.find_best_arm_path()
        nodes_path = [self.root]
        node = self.root
        for idx in self.best_strategy_path:
            nodes_path.append(node.children[idx])
            node = node.children[idx]
        self.best_strategy_nodes_path = nodes_path

    def get_reward_by_node(self, node):
        """
        Calculates the reward for a given node based on its value and level.

        Parameters
        ----------
        node : Node
            The node for which the reward is calculated.

        Returns
        -------
        float
            The reward for the given node.
        """
        level_correction = 10 ** (-node.level)
        h = 0.1 * level_correction
        low = node.value - h / 2
        high = node.value + h / 2
        return self.sampling_rng.uniform(low=low, high=high)

    def get_R_level(self, level):
        """
        Computes a correction factor based on the level of the node.

        Parameters
        ----------
        level : int
            The level for which the correction factor is computed.

        Returns
        -------
        float
            The correction factor for the given level.
        """
        return 10 * 10 ** (-level)

    def get_reward_by_path(self, path):
        """
        Calculates the total reward for a given path by summing up the rewards of the nodes.

        Parameters
        ----------
        path : list
            A list of nodes representing the path.

        Returns
        -------
        float
            The total reward for the given path.
        """
        path_wo_root = path[1:]  # Exclude root node
        return np.sum([self.get_reward_by_node(node) for node in path_wo_root])

    def get_best_strategy_reward(self):
        """
        Returns the reward associated with the best strategy (i.e., the best path).

        Returns
        -------
        float
            The reward for the best strategy.
        """
        return self.get_reward_by_path(self.best_strategy_nodes_path)

    def create_sub_level(self, node_key):
        """
        Creates a sub-level for the given node by generating children nodes.
        
        Parameters
        ----------
        node_key : str
            The key of the node for which to create a sub-level.
        """
        node = self.tree.graph[node_key]
        level = node.level + 1
        
        # Change the number of children based on the distribution strategy
        if self.node_distribution == 'random':
            num_children = self.env_rng.choice(range(1, 11))  # Random between 1 and 10
        elif self.node_distribution == 'imbalanced':
            num_children = self.env_rng.choice([2, 6])  # Fewer nodes or more nodes
        else:  # 'balanced'
            num_children = self.nb_leaves_per_class  # Fixed number of children per node

        for n in range(num_children):
            name = 'level_{}_child_number_{}_of_{}'.format(level, n, node.name)
            value = self.env_rng.choice(range(1, 9))
            data = name, value
            _, node = self.tree.insert(node, data)

    def iterative_graph_create(self, start=None):
        """
        Creates the tree iteratively by traversing the graph using breadth-first search (BFS).
        
        Parameters
        ----------
        start : str, optional
            The starting node for the tree creation. Defaults to 'root'.

        Returns
        -------
        visited : list
            A list of nodes visited during the BFS traversal.
        """
        if not start:
            start = 'root'

        visited = []
        queue = deque()
        queue.append(start)

        while queue:
            node_key = queue.popleft()
            if node_key not in visited:
                visited.append(node_key)
                self.create_sub_level(node_key)
                unvisited = [n.name for n in self.tree.graph[node_key].children if
                             (n.name not in visited and n.level < self.nb_levels)]
                queue.extend(unvisited)

        return visited
    

    def visualize_tree(self):
        """
        Visualizes the tree structure using a graph plot. Each node's name, value, and level
        are displayed. The edges represent the relationships between parent and child nodes.

        The visualization uses `networkx` for graph creation and `matplotlib` for plotting.
        """
        G = nx.DiGraph()

        def add_nodes_and_edges(node, parent=None, group=0):
            reward = self.get_reward_by_node(node)
            G.add_node(node.name, level=node.level, value=node.value, reward=reward, group=group)

            for idx, child in enumerate(node.children):
                add_nodes_and_edges(child, node, group=idx+1)  # Chaque groupe d'enfants d'un parent a un index unique
                G.add_edge(node.name, child.name)

        add_nodes_and_edges(self.root)

        pos = {}
        for node in G.nodes():
            level = G.nodes[node]['level']
            pos[node] = (level, -len(pos))

        plt.figure(figsize=(10, 8))

        labels = {}
        for node in G.nodes():
            level = G.nodes[node]['level']
            group = G.nodes[node]['group']
            reward = G.nodes[node]['reward']

            parent_nodes = list(G.predecessors(node))
            if parent_nodes:
                parent_node = parent_nodes[0]
                child_index = list(G.neighbors(parent_node)).index(node) + 1
            else:
                child_index = 0

            if level == 0:
                node_label = f"Level {level}\nGroup {group}\nReward: {reward:.2f}"
            else:
                node_label = f"Level {level}\nGroup {group}\nChild n° {child_index}\nReward: {reward:.2f}"

            labels[node] = node_label

        nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

        plt.title("Tree Visualization")
        plt.show()



""" 
class PolarizedEnvironment(Environment):
    def __init__(self, rd, nb_leaves_per_class, nb_levels, node_distribution):
        super().__init__(rd, nb_leaves_per_class, nb_levels, node_distribution)
        self.nb_levels = 2  # Fix the number of levels to 2.

    def create_sub_level(self, node_key):
        node = self.tree.graph[node_key]
        level = node.level + 1
        self.side = 1

        if level == 1:
            for n in range(1):  # Groupe 1 (1 nœud)
                name = f'level_{level}_group_1_child_{n+1}_of_{node.name}'  
                value = self.env_rng.choice(range(1, 9))
                data = name, value
                _, node = self.tree.insert(node, data)  # Insérer comme enfant

            for n in range(1):  # Groupe 2 (1 nœud)
                name = f'level_{level}_group_2_child_{n+1}_of_{node.name}'  
                value = self.env_rng.choice(range(1, 9))  
                data = name, value
                _, node = self.tree.insert(node, data)

        elif level == 2:
            if self.side == 1:
                for n in range(2):  # Groupe 1 (2 nœuds)
                    name = f'level_{level}_group_1_child_{n+1}_of_{node.name}'  
                    value = self.env_rng.choice(range(8, 20))  # Gros reward
                    data = name, value
                    _, node = self.tree.insert(node, data)
                    self.side += 1
            elif self.side == 2:
                for n in range(8):  # Groupe 2 (8 nœuds)
                    name = f'level_{level}_group_2_child_{n+1}_of_{node.name}'  
                    value = self.env_rng.choice(range(1, 5))  # Faible reward
                    data = name, value
                    _, node = self.tree.insert(node, data)

    def iterative_graph_create(self, start=None):
        if not start:
            start = 'root'

        visited = []
        queue = deque()
        queue.append(start)

        while queue:
            node_key = queue.popleft()
            if node_key not in visited:
                visited.append(node_key)
                self.create_sub_level(node_key)  
                unvisited = [n.name for n in self.tree.graph[node_key].children if 
                             (n.name not in visited and n.level < self.nb_levels)]
                queue.extend(unvisited)

        return visited

    def set(self):
        self.iterative_graph_create()  # Create the tree iteratively.
        _, self.best_strategy_path = self.tree.find_best_arm_path()  # Find the best strategy.

        nodes_path = [self.root]  # Initialize the best strategy path.
        node = self.root
        for idx in self.best_strategy_path:
            nodes_path.append(node.children[idx])  # Add the child node to the strategy path.
            node = node.children[idx]
        self.best_strategy_nodes_path = nodes_path  # Store the best strategy.

    def get_reward_by_node(self, node):
        level_correction = 10 ** (-node.level)
        h = 0.1 * level_correction

        if node.level == 1:
            low = node.value - h / 2
            high = node.value + h / 2
        elif node.level == 2:
            if "group_1" in node.name:
                low = node.value - h / 2
                high = node.value + h / 2
            else:  # Group 2 gets lower rewards.
                low = node.value - h
                high = node.value + h
        else:
            low = 0
            high = 1

        return self.sampling_rng.uniform(low=low, high=high)  # Sample a reward in the calculated range.

    def visualize_tree(self):
        G = nx.DiGraph()

        def add_nodes_and_edges(node, parent=None, group=0):
            reward = self.get_reward_by_node(node)
            G.add_node(node.name, level=node.level, value=node.value, reward=reward, group=group)
            for idx, child in enumerate(node.children):
                G.add_edge(node.name, child.name)
                add_nodes_and_edges(child, node, group=idx+1)

        add_nodes_and_edges(self.root)

        pos = {}
        for node in G.nodes():
            level = G.nodes[node]['level']
            pos[node] = (level, -len(pos))

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

        labels = {node: f"Level {G.nodes[node]['level']}\nGroup {G.nodes[node]['group']}\nReward: {G.nodes[node]['reward']:.2f}"
                  for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

        plt.title("Tree Visualization")
        plt.show()
"""
