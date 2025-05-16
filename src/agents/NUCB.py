import numpy as np 

class NUCB:
  def __init__(self, tree):
    self.tree = tree
    self.tree_structure = self.tree.extract_tree_structure_from_tree()
    self.L = len(self.tree_structure)
    leaves = self.tree.get_all_leaves()
    self.paths = [self.tree.get_parent_nodes(leaf) for leaf in leaves]
    self.len_paths = len(self.paths)
    self.all_nodes = tree.get_all_nodes()
    self.reset()     

  def reset(self, seed=None):
    self.rng = np.random.RandomState(seed) 
    self.total_reward = 0
    self.t = 0  
    self.counts = {node.name: float(0) for node in self.all_nodes}
    self.rewards = {node.name: float(0) for node in self.all_nodes}
  
  def get_path(self):
    current_node = self.tree_structure[0][0]  # root
    path = [current_node]

    if self.t < self.len_paths:  # Exploration phase
        path = self.paths[self.t % self.len_paths]
    else:
        for _ in self.tree_structure[1:]:
            parent_node = path[-1]
            children = parent_node.children  # Only consider children of the last chosen node

            # Filter out children with 0 visits to avoid division by zero
            count_rewards = np.array([self.rewards[child.name] for child in children])
            count_actions = np.array([self.counts[child.name] for child in children])

            # Avoid division by zero in empirical means
            empirical_means = np.zeros_like(count_rewards)
            mask = count_actions > 0
            empirical_means[mask] = count_rewards[mask] / count_actions[mask]

            # UCB term: infinite if not yet visited
            ucbs = np.sqrt(2 * np.log(self.t) / np.maximum(count_actions, 1e-6))
            ucbs[~mask] = float('inf')  # Ensure unvisited nodes are explored

            scores = empirical_means + ucbs
            best_child_idx = np.argmax(scores)
            node_action = children[best_child_idx]

            path.append(node_action)
            self.counts[node_action.name] += 1

    self.t += 1
    self.tree.step()
    return path

  def receive_reward(self, reward, path_reward, path):
    self.total_reward += reward
    for node, reward in zip(path, path_reward):
      self.rewards[node.name] += reward
      self.counts[node.name] += 1

  def nested(self):
    return True

  def name(self):
    return 'NUCB' 