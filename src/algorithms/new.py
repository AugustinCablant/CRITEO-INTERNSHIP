import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np
from tqdm import tqdm
import time
import psutil
from src.utils.save_results import save_result

# Small constant to prevent division by zero errors
EPS = 1e-8

class NestedExponentialWeights:
    """
    A class implementing the Nested Exponential Weights (NEWE) algorithm for decision-making
    in environments with rewards and losses over a series of rounds.

    This algorithm is designed for settings where an agent must make sequential decisions 
    from a set of actions, each of which may depend on previously chosen actions (e.g., tree-based models).

    Attributes:
    -----------
    rng : np.random.RandomState
        The random number generator used for sampling actions.
    max_round : int
        The maximum number of rounds for which the algorithm will run.
    settings : dict
        A dictionary containing the settings for the algorithm, including random seed, 
        number of rounds, etc.
    environment : function
        A function representing the environment that returns rewards/losses for chosen actions.
    """

    def __init__(self, rd, max_rounds):
        self.rng = np.random.RandomState(rd)  
        self.max_round = max_rounds           

    def set_environment(self, environment):
        self.environment = environment

    def vector_proba(self, y):
        stable_exp_y = np.exp(y - np.max(y))  
        proba_vector = stable_exp_y / np.sum(stable_exp_y)  
        return proba_vector

    def sample_node_path(self, round):
        node_path = []  # Path of nodes chosen
        proba_path = []  # Corresponding probabilities for each node
        reward_path = []  # Corresponding rewards for each node
        node = self.environment.tree.root  # Start from the root node

        while bool(node.children):  # Continue as long as the current node has children
            lr = 1 / np.sqrt(round + 1)  # Learning rate decreases over time
            proba = self.vector_proba(node.scores_children * lr)  # Get probabilities using softmax
            idx_list = range(node.nb_children)  # List of child indices
            idx_node = self.rng.choice(idx_list, p=proba)  # Sample a child based on the probabilities
            child_node = node.children[idx_node]  # Select the chosen child
            node_path.append(idx_node)  # Append the index to the path
            reward_child = self.environment.get_reward_by_node(child_node)  # Get reward for the chosen child
            reward_path.append(reward_child)  # Append the reward
            proba_path.append(proba[idx_node])  # Append the probability
            node = child_node  # Move to the child node for the next iteration

        return node_path, proba_path, reward_path

    def update_score(self, nodes_path, proba_path, reward_path):
        node = self.environment.tree.root  # Start from the root node
        proba = 1  # Initial probability is 1

        # Iterate through the nodes in the path, updating scores based on rewards
        for idx_node, P, reward in zip(nodes_path, proba_path, reward_path):
            proba *= P  # Update the joint probability
            node.scores_children[idx_node] = node.scores_children[idx_node] + reward / (proba + EPS)
            node = node.children[idx_node]  # Move to the next node in the path

    def iterate_learning(self):
        metrics = {
            'reward': [],
            'regret': [],
            'round': []
        }
        regrets = []  # List to track regrets
        rewards = []  # List to track rewards

        # Iterate over all rounds
        for round in tqdm(range(0, self.max_round)):

            # Sample a path through the tree, get rewards, and update metrics
            node_path, proba_path, reward_path = self.sample_node_path(round)

            # Calculate the total reward and regret for the round
            reward = np.sum(reward_path)
            best_strategy_reward = self.environment.get_best_strategy_reward()
            regrets.append(best_strategy_reward - reward)  # Calculate regret
            rewards.append(reward)  # Add reward to list

            # Update the scores of the nodes along the path
            self.update_score(node_path, proba_path, reward_path)

            # Every 100 rounds, save and log the metrics
            if round % 100 == 0:
                metrics['reward'].append(np.mean(rewards))  # Average reward
                regret = np.sum(regrets)  # Total regret
                metrics['regret'].append(regret)
                metrics['round'].append(round)
                save_result(self.settings, regret, np.mean(rewards), round)

        # Final visualization (if any)
        self.score_vector = None  # Not used but could be extended for visualizing score distribution
        return metrics
    
    def name(self):
      return 'NEW' 
