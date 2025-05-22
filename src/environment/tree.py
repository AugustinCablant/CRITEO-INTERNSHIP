import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt 

class Node:
    def __init__(self, name, parent=None, mean=0, var=1, seed=None):
        #self.seed = seed
        #self.rng =  np.random.RandomState(seed)
        self.rng = np.random.default_rng()
        self.name = name
        self.mean = mean  
        self.var = var  
        self.children = []
        self.scores_children = np.array([])
        self.nb_children = 0
        self.parent = parent
        self.level = parent.level + 1 if parent else 0
        self.reset()
    

    def get_child_nodes(self):
        if not nodes:
            nodes = []
        for child in self.children:
            nodes.append((child.name, child.level, child.nb_children))
            nodes.extend(child.get_child_nodes())
        return nodes

    def reset(self):
        value = self.mean + np.sqrt(self.var) * self.rng.normal()
        self.value = value
        return value

class NodeUniform:
    def __init__(self, name, parent=None, seed=None, best=False):
        #self.seed = seed
        self.best = best
        #self.rng =  np.random.RandomState(seed)
        self.rng = np.random.default_rng()
        self.name = name
        self.children = []
        self.scores_children = np.array([])
        self.nb_children = 0
        self.parent = parent
        self.level = parent.level + 1 if parent else 0
        self.reset()
    

    def get_child_nodes(self):
        if not nodes:
            nodes = []
        for child in self.children:
            nodes.append((child.name, child.level, child.nb_children))
            nodes.extend(child.get_child_nodes())
        return nodes

    def reset(self):
        if self.best:
            value = 1
            self.value = value
        else:
            value = self.rng.uniform()
            self.value = value
        return value


class Tree:
    def __init__(self, seed=None, uniform=False):
        self.uniform = uniform
        self.levels = [[]]
        self.graph = {'root': None,}
        self.max_level = 0
        #self.seed = seed
        self.rng = np.random.RandomState(seed)

    def step(self):
        for key, node in self.graph.items():
            node.reset()
            self.graph[key] = node

    def create_node(self, name, parent=None, mean=0, var=1, best=False):
        if self.uniform:
            return NodeUniform(name, parent, best)
        else:
            return Node(name, parent, mean, var)  ##

    def insert(self, parent_node, name, mean=0, var=1, best=False):
        if parent_node is None:
            if self.uniform:
                node = self.create_node(name, mean=mean, var=var, best=best)
            else:
                node = self.create_node(name, mean=mean, var=var)
            if node.level == 0:
                self.root = node
                value = node.reset()
                self.graph['root'] = node
            return node
        if self.uniform:
            node = self.create_node(name, parent_node, mean, var, best)
        else:
            node = self.create_node(name, parent_node, mean, var)
        value = node.reset()
        self.graph[name] = node
        parent_node.children.append(node)
        parent_node.nb_children = len(parent_node.children)
        parent_node.scores_children = np.full(parent_node.nb_children, 1.0 / parent_node.nb_children)
        self.max_level = max(self.max_level, node.level)
        return node, parent_node

    def get_parent_nodes(self, node):
        nodes = [node]
        def _recursive_parent_nodes(node, nodes):
            if node.parent:
                nodes.append(node.parent)
                _recursive_parent_nodes(node.parent, nodes)
            return nodes
        nodes = _recursive_parent_nodes(node, nodes)
        nodes.reverse()
        return nodes

    def get_all_nodes(self):
        nodes = [self.root]
        def _get_nodes(node):
            for child in node.children:
                nodes.append(child)
                _get_nodes(child)
        _get_nodes(self.root)
        nodes = sorted(nodes, key=lambda node: node.level)
        nodes_names = [node.name for node in nodes]
        return nodes

    def get_all_leaves(self):
        leaves = []
        def get_leaves(node):
            if len(node.children) == 0:
                leaves.append(node)
            if node is not None:
                for child in node.children:
                    get_leaves(child)
        get_leaves(self.root)
        return leaves

    def get_reward_leaf(self, leaf):
        parents_leaf = self.get_parent_nodes(leaf)
        reward = 0 
        for node in parents_leaf:
            reward += node.value
        return reward
    
    def get_mu_leaf(self, leaf):
        parents_leaf = self.get_parent_nodes(leaf)
        mu = 0 
        for node in parents_leaf:
            mu += node.mean
        return mu
    
    def get_reward_leaves(self):
        leaves = self.get_all_leaves()
        data = []
        for leaf in leaves:
            data.append([leaf, leaf.name, self.get_reward_leaf(leaf)])
        return data

    def get_mu_leaves(self):
        leaves = self.get_all_leaves()
        data = []
        for leaf in leaves:
            data.append([leaf, leaf.name, self.get_mu_leaf(leaf)])
        return data


    def find_best_arm_path(self):
        data = self.get_mu_leaves()
        best_leaf_index = np.argmax([x[2] for x in data])
        best_leaf, _, _ = data[best_leaf_index]
        path_nodes = self.get_parent_nodes(best_leaf)
        #path_rewards = [node.value for node in path_nodes]
        #path_names = [node.name for node in path_nodes]
        return path_nodes
    
    def extract_tree_structure_from_tree(self):
        structure = []
        all_nodes = self.get_all_nodes()
        max_level = self.max_level
        
        for level in range(max_level + 1):
            level_nodes = [node for node in all_nodes if node.level == level]
            structure.append(level_nodes)
            
        return structure

    
    def visualize_tree_mu(self):
        G = nx.DiGraph()
        labels = {}
        edge_labels = {}
        pos = {}
        node_colors = []

        def add_edges(node, pos_x=0, pos_y=0, layer_width=1.0):
            node_id = id(node)
            G.add_node(node_id)
            labels[node_id] = f"{node.name}\nμ={node.mean:.2f}"
            pos[node_id] = (pos_x, -pos_y)
            node_colors.append(node.level)

            num_children = len(node.children)
            width_step = layer_width / max(num_children, 1)

            for i, child in enumerate(node.children):
                child_id = id(child)
                G.add_edge(node_id, child_id)
                edge_labels[(node_id, child_id)] = f"μ = {child.mean:.2f}"

                child_x = pos_x - layer_width / 2 + (i + 0.5) * width_step
                add_edges(child, child_x, pos_y + 1, width_step)

        add_edges(self.root)

        cmap = plt.cm.viridis
        colors = [cmap(l / (self.max_level + 1)) for l in node_colors]

        plt.figure(figsize=(14, 8))
        nx.draw(
            G, pos, labels=labels, node_color=colors,
            with_labels=True, node_size=4000, font_size=9, font_color='white'
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=self.max_level))
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=range(self.max_level + 1))
        cbar.ax.set_yticklabels([f"Level {i}" for i in range(self.max_level + 1)])
        plt.title("Tree Structure with μ per Node")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    def visualize_tree(self, t):
        G = nx.DiGraph()
        labels = {}
        edge_labels = {}
        pos = {}
        node_colors = []

        def add_edges(node, pos_x=0, pos_y=0, layer_width=1.0):
            G.add_node(node.name)
            labels[node.name] = node.name
            pos[node.name] = (pos_x, -pos_y)
            node_colors.append(node.level)

            num_children = len(node.children)
            width_step = layer_width / max(num_children, 1)

            for i, child in enumerate(node.children):
                reward = child.value
                G.add_edge(node.name, child.name)
                edge_labels[(node.name, child.name)] = f"{reward:.2f}"

                child_x = pos_x - layer_width / 2 + (i + 0.5) * width_step
                add_edges(child, child_x, pos_y + 1, width_step)

        add_edges(self.root)
        for node in self.get_all_nodes():
            if len(node.children) == 0:  
                total_reward = self.get_reward_leaf(node)
                labels[node.name] = f"{node.name}\n({total_reward:.2f})"

        cmap = plt.cm.viridis
        colors = [cmap(l / (self.max_level + 1)) for l in node_colors]
        plt.figure(figsize=(14, 8))
        nx.draw(
            G, pos, labels=labels, node_color=colors,
            with_labels=True, node_size=5000, font_size=10, font_color='white'
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=self.max_level))
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=range(self.max_level + 1))
        cbar.ax.set_yticklabels([f"Level {i}" for i in list(range(self.max_level + 1))[::-1]])
        plt.title(f"Simulation at time t = {t}")
        plt.axis('off')
        plt.show()