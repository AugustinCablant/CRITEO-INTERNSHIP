import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt 

rng = np.random.RandomState(2025)


class Node:
    def __init__(self, data, parent=None, mean=0, var=1):
        self.name, self.value = data
        self.mean = mean  
        self.var = var  
        self.children = []
        self.scores_children = np.array([])
        self.nb_children = 0
        self.parent = parent
        self.level = parent.level + 1 if parent else 0
        self.value = self.value * (10 ** (-self.level))  

    def get_child_nodes(self):
        if not nodes:
            nodes = []
        for child in self.children:
            nodes.append((child.name, child.value, child.level, child.nb_children))
            nodes.extend(child.get_child_nodes())
        return nodes

    def get_reward(self):
        """Generate the reward"""
        return self.mean + np.sqrt(self.var) * np.random.normal()



class Tree:
    """
    Class tree will provide a tree as well as utility functions.
    """

    def __init__(self):
        """
        """
        self.levels = [[]]
        self.graph = {
            'root': None,
        }
        self.max_level = 0

    def create_node(self, data, parent=None, mean=0, var=1):
        """
        Utility function to create a node.
        """
        return Node(data, parent, mean, var)

    def insert(self, parent_node, data):
        """
        Insert function will insert a node into tree.
        Duplicate keys are not allowed.
        """
        # if tree is empty , return a root node
        name, value = data
        if parent_node is None:
            node = self.create_node(data)
            if node.level == 0:
                self.root = node
                self.graph['root'] = node
            return node

        node = self.create_node(data, parent_node)
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
        return [(node.name, node.data) for node in nodes]

    def get_all_nodes(self):
        """ Get all node utility function

        Returns all nodes of the graph
        -------

        """
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
        """

        Returns
        -------

        """

        leaves = []

        def get_leaves(node):
            # print(node.name, node.children)
            if len(node.children) == 0:
                leaves.append(node)
            if node is not None:
                for child in node.children:
                    get_leaves(child)

        get_leaves(self.root)

        return leaves


    def iterative_dfs(self, node_key=None):
        """" Depth-first search

        Parameters
        ----------
        node

        Returns
        -------

        """
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
        """
        Breadth-First Search
        Parameters
        ----------
        start

        Returns
        -------

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
                unvisited = [n.name for n in self.graph[node_key].children if n.name not in visited]
                queue.extend(unvisited)

        return visited

    def find_max_sum_path(self, root, max_result=-np.infty, max_path=[]):

        # Base Case
        if root is None:
            return 0, max_result

        max_sums = [0]
        max_paths = [[]]
        for child in root.children:
            max_sum, idx_max_path = self.find_max_sum_path(child, max_result, max_path)
            max_sums.append(max_sum)
            max_paths.append(idx_max_path)

        sums = root.value + np.array(max_sums)
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
    

    