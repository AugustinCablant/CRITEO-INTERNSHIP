import numpy as np

class ExploreThenCommit:
    def __init__(self, action_set, exploration_rounds=5):
        self.action_set = action_set
        self.K = len(self.action_set)
        self.exploration_rounds = exploration_rounds  # Number of times each arm is explored
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.total_reward = 0
        self.t = 0
        self.count_actions = np.zeros(self.K)
        self.count_rewards = np.zeros(self.K)
        self.committed = False
        self.best_action = None

    def get_action(self):
        if not self.committed:
            # Still exploring
            action = self.t % self.K
            round_number = self.t // self.K
            if round_number >= self.exploration_rounds - 1 and action == self.K - 1:
                # End of exploration phase
                self.committed = True
                empirical_means = self.count_rewards / self.count_actions
                self.best_action = int(np.argmax(empirical_means))
        else:
            action = self.best_action

        self.t += 1
        self.count_actions[action] += 1
        return self.action_set[action], action

    def receive_reward(self, reward, action_index):
        self.total_reward += reward
        self.count_rewards[action_index] += reward

    def nested(self):
        return False

    def name(self):
        return f"ETC({self.exploration_rounds})"