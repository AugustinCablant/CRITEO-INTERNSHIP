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
        """
        Initializes the NEW agent with the given settings.

        Parameters:
        -----------
        settings : dict
            A dictionary containing the settings for the agent:
            - 'rd': random seed for random number generation.
            - 'max_rounds': maximum number of rounds for the algorithm to run.
        """
        self.rng = np.random.RandomState(rd)  # Initialize random state
        self.max_round = max_rounds           # Maximum number of rounds

    def set_environment(self, environment):
        """
        Sets the environment function which the agent will interact with.

        The environment function should take two arguments:
        - A vector of size K representing the chosen slate (where non-zero entries indicate 
          the chosen actions).
        - The current round number, t.

        It should return the rewards/losses for the chosen slate, with rewards clipped to 
        be in the range [-1, 1]. Non-chosen actions in the slate should return 0.

        Parameters:
        -----------
        environment : function
            A function that takes a vector of actions (slate) and the current round number as inputs,
            and returns the reward/loss associated with the chosen actions.
        """
        self.environment = environment

    def vector_proba(self, y):
        """
        Converts a vector of values into a probability distribution using the softmax function.

        This is done in a numerically stable way to avoid overflow issues.

        Parameters:
        -----------
        y : np.ndarray
            A vector of values (usually scores or preferences for actions).

        Returns:
        --------
        np.ndarray
            A vector of probabilities corresponding to the input values, where the probabilities 
            sum to 1.
        """
        stable_exp_y = np.exp(y - np.max(y))  # Shift y to avoid overflow
        proba_vector = stable_exp_y / np.sum(stable_exp_y)  # Normalize to get probabilities
        return proba_vector

    def sample_node_path(self, round):
        """
        Samples a path through the decision tree, starting from the root.

        In each round, the agent chooses an action based on the current scores of the nodes 
        (actions), and accumulates rewards.

        Parameters:
        -----------
        round : int
            The current round number.

        Returns:
        --------
        tuple
            A tuple (node_path, proba_path, reward_path), where:
            - node_path : A list of node indices representing the path taken.
            - proba_path : A list of probabilities associated with each node along the path.
            - reward_path : A list of rewards received at each node along the path.
        """
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
        """
        Updates the scores of the nodes along the path based on the rewards received.

        The score update is performed using a weighted average, where the weight depends 
        on the probability of selecting the node.

        Parameters:
        -----------
        nodes_path : list
            A list of node indices representing the path taken.
        proba_path : list
            A list of probabilities associated with each node along the path.
        reward_path : list
            A list of rewards received at each node along the path.
        """
        node = self.environment.tree.root  # Start from the root node
        proba = 1  # Initial probability is 1

        # Iterate through the nodes in the path, updating scores based on rewards
        for idx_node, P, reward in zip(nodes_path, proba_path, reward_path):
            proba *= P  # Update the joint probability
            node.scores_children[idx_node] = node.scores_children[idx_node] + reward / (proba + EPS)
            node = node.children[idx_node]  # Move to the next node in the path

    def iterate_learning(self):
        """
        Runs the agent for a set number of rounds, making decisions and updating scores.
        
        During each round, the agent samples a node path, receives rewards, and updates 
        the node scores. The agent's performance metrics are tracked, including regret 
        and total reward.

        The function also periodically saves results and visualizes the progress.

        Returns:
        --------
        dict
            A dictionary containing the performance metrics over all rounds:
            - 'reward': List of average rewards at each checkpoint.
            - 'regret': List of total regrets at each checkpoint.
            - 'round': List of round numbers at which metrics were recorded.
        """
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
