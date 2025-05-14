import numpy as np 

class UCB:
  def __init__(self, action_set):
        self.action_set = action_set
        self.K = len(self.action_set)
        self.reset()     

  def reset(self, seed=None):
      self.rng = np.random.RandomState(seed) 
      self.total_reward = 0
      self.t = 0  
      self.count_actions = np.zeros(self.K)
      self.count_rewards = np.zeros(self.K)
  
  def get_action(self):
    if self.t < 2 * self.K:
        action = self.t % self.K
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
                empirical_means = np.divide(self.count_rewards, self.count_actions, out=np.zeros_like(self.count_rewards), where=self.count_actions != 0)
                ucbs = np.sqrt(2 * np.log(self.t) / self.count_actions)
                scores = empirical_means + ucbs
                action = np.argmax(scores)

    self.t += 1
    self.count_actions[action] += 1
    return self.action_set[action], action

  def receive_reward(self, reward, action_index):
    self.total_reward += reward
    self.count_rewards[action_index] += reward

  def nested(self):
      return False

  def name(self):
    return 'UCB' 