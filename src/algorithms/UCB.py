import numpy as np 

class UCB:
  def __init__(self, arms, rd):
    self.rng = np.random.RandomState(rd) 
    self.arms = arms
    self.K = len(arms)
    self.count_actions = np.zeros(self.K)
    self.count_rewards = np.zeros(self.K)
    self.t = 0
  
  def get_action(self, action_set):
    if self.t < self.K:
        action = self.t
    else:
        empirical_means = self.count_rewards / self.count_actions
        ucbs = np.sqrt(2 * np.log(self.t) / self.count_actions) 
        action = np.argmax(empirical_means + ucbs)

    self.t += 1
    self.count_actions[action] += 1
    self.current_action = action #need to remember the *index* of the action now
    return action_set[action]

  def receive_reward(self, action, reward):
    self.count_rewards[self.current_action] += reward

  def reset(self):
    self.count_actions = np.zeros(self.K)
    self.count_rewards = np.zeros(self.K)
    self.t = 0

  def compute_condition(self):
    empirical_means = self.count_rewards / self.count_actions
    star_hat = np.argmax(empirical_means)
    mu_star_hat = empirical_means[star_hat]
    N_star = self.count_actions[star_hat]
    inv_N_star = 1 / N_star
    inf = np.min([((mu_star_hat - empirical_means[k])**2)/(inv_N_star + (1/self.count_actions[k])) for k in list(range(self.K)) if k!=star_hat])
    
    return inf

  def name(self):
    return 'UCB' 