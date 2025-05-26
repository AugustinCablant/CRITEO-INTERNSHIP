import gym 
from src.environment.tree import Tree
import numpy as np 

class EnvironmentExample(gym.Env):
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.tree = None
        self.reset()
        
    def reset(self):
        self.tree = self.generate_tree()
    
    def generate_tree(self):  
        tree = Tree(uniform=False, seed=self.seed) 
        root = tree.insert(parent_node=None, name="Targeting", mean=0, var=0)

        # First Layer
        families, _ = tree.insert(parent_node=root, name="families", mean=0.20, var=0.5)
        professionals, _ = tree.insert(parent_node=root, name="professionals", mean=0.8, var =0.5)

        # Second Layer
        adults, _ = tree.insert(parent_node=families, name="adults", mean=0.1, var=0.2)
        youngs, _ = tree.insert(parent_node=families, name="youngs", mean=0.1, var=0.2)

        start_up, _ = tree.insert(parent_node=professionals, name="start-up", mean=0.2, var=0.2)
        companies, _ = tree.insert(parent_node=professionals, name="companies", mean=0.4, var=0.2)

        # Third Layer
        radio, _ = tree.insert(parent_node=adults, name="radio", mean=0.01, var=0.1)
        tv, _ = tree.insert(parent_node=adults, name="tv", mean=0.01, var=0.1)
        magazines, _ = tree.insert(parent_node=adults, name="magazines", mean=0.01, var=0.1)
        tv, _ = tree.insert(parent_node=adults, name="supermarket", mean=0.05, var=0.1)

        youtube, _ = tree.insert(parent_node=youngs, name="youtube", mean=0.06, var=0.1)
        social_networks, _ = tree.insert(parent_node=youngs, name="social networks", mean=0.01, var=0.1)
        
        webinaire, _ = tree.insert(parent_node=start_up, name="webinaire", mean=0.09, var=0.1)
        events, _ = tree.insert(parent_node=start_up, name="events", mean=0.1, var=0.1)

        linkedin, _ = tree.insert(parent_node=companies, name="linkedin", mean=0.1, var=0.1)
        email, _ = tree.insert(parent_node=companies, name="email", mean=0.01, var=0.1)

        tree.step()
        return tree
    
    
    def get_action_set(self):
        return self.tree.get_all_leaves() #
    
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
        return np.sum([node.value for node in path]), [node.value for node in path], np.sum([node.mean for node in path])
    
    def get_total_reward_per_leaf(self):
        action_set = self.get_action_set()
        total_reward_per_leaf = [self.get_reward(i) for i, leaf in enumerate(action_set)]
        self.total_reward_per_leaf = total_reward_per_leaf
        return total_reward_per_leaf
    
    def get_best_strategy_reward(self):
        best_arm_path =  self.tree.find_best_arm_path()
        return [node.name for node in best_arm_path] , np.sum([node.value for node in best_arm_path])
    
    def step(self, seed):
        self.tree = self.generate_tree(seed)
    
    def render(self, mode='human', close=False):
        pass