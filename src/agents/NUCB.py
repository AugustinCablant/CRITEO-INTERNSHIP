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
    path = []
    path.append(current_node)

    if self.t < self.len_paths: # Explore
       path = self.paths[self.t % self.len_paths]
    else:
        for layer in self.tree_structure[1:]:
           count_rewards = np.array([self.rewards[node.name] for node in layer])
           count_actions = np.array([self.counts[node.name] for node in layer])
           empirical_means = count_rewards / count_actions
           ucbs = np.sqrt(2 * np.log(self.t) / count_actions)
           scores = empirical_means + ucbs
           layer_action = np.argmax(scores)
           node_action = layer[layer_action]
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