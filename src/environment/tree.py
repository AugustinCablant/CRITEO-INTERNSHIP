import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt 

rng = np.random.RandomState(2025)


class Node:
    def __init__(self, name, parent=None, mean=0, var=1):
        self.name = name
        self.mean = mean  
        self.var = var  
        self.children = []
        self.scores_children = np.array([])
        self.nb_children = 0
        self.parent = parent
        self.level = parent.level + 1 if parent else 0

    def get_child_nodes(self):
        if not nodes:
            nodes = []
        for child in self.children:
            nodes.append((child.name, child.level, child.nb_children))
            nodes.extend(child.get_child_nodes())
        return nodes

    def get_reward(self):
        return self.mean + np.sqrt(self.var) * np.random.normal()



class Tree:
    def __init__(self):
        self.levels = [[]]
        self.graph = {'root': None,}
        self.max_level = 0

    def create_node(self, name, parent=None, mean=0, var=1):
        return Node(name, parent, mean, var)

    def insert(self, parent_node, name, mean, var):
        if parent_node is None:
            node = self.create_node(name, mean=mean, var=var)
            if node.level == 0:
                self.root = node
                self.graph['root'] = node
            return node

        node = self.create_node(name, parent_node, mean, var)
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
            reward += node.get_reward()
        return reward
    
    def get_reward_leaves(self):
        leaves = self.get_all_leaves()
        data = []
        for leaf in leaves:
            data.append([leaf.name, self.get_reward_leaf(leaf)])
        return data


    def iterative_dfs(self, node_key=None):
        if not node_key:
            node_key = 'root'
        visited = []
        stack = deque()
        stack.append(node_key)
        while stack:
            node_key = stack.pop()
            if node_key not in visited:
                visited.append(node_key)
                unvisited = [n.name for n in self.graph[node_key].children if n.name not in visited]
                stack.extend(unvisited)
        return visited

    def iterative_bfs(self, start=None):
        if not start:
            start = 'root'
        visited = []
        queue = deque()
        queue.append(start)
        while queue:
            node_key = queue.popleft()
            if node_key not in visited:
                visited.append(node_key)
                unvisited = [n.name for n in self.graph[node_key].children if n.name not in visited]
                queue.extend(unvisited)
        return visited

    def find_max_sum_path(self, root, max_result=-np.infty, max_path=[]):
        if root is None:
            return 0, max_result
        max_sums = [0]
        max_paths = [[]]
        for child in root.children:
            max_sum, idx_max_path = self.find_max_sum_path(child, max_result, max_path)
            max_sums.append(max_sum)
            max_paths.append(idx_max_path)
        sums = root.get_reward() + np.array(max_sums)
        idx = np.argmax(sums)
        max_result = sums[idx]
        max_path = max_paths[idx]
        max_path.append(idx)
        return max_result, max_path

    def find_best_arm_path(self):
        max_mean, path = self.find_max_sum_path(self.root)
        path = np.array(path[1:])
        path = list(path - 1)[::-1]
        return max_mean, path
    
    def visualize_tree(self):
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
                reward = child.get_reward()
                G.add_edge(node.name, child.name)
                edge_labels[(node.name, child.name)] = f"{reward:.2f}"

                child_x = pos_x - layer_width/2 + (i + 0.5) * width_step
                add_edges(child, child_x, pos_y + 1, width_step)

        add_edges(self.root)
        cmap = plt.cm.viridis
        colors = [cmap(l / (self.max_level + 1)) for l in node_colors]
        plt.figure(figsize=(14, 8))
        nx.draw(G, pos, labels=labels, node_color=colors, with_labels=True, node_size=2000, font_size=10, font_color='white')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=self.max_level))
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=range(self.max_level + 1))
        cbar.ax.set_yticklabels([f"Level {i}" for i in range(self.max_level + 1)])
        plt.title("Environment")
        plt.axis('off')
        plt.show()
    

    