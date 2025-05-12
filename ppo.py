

import gymnasium as gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
	""" Implementation of PPO using Actor-Critic 
	The actor takes the decisions while the critic evaluate the quality of these decisions."""
	def __init__(self, policy_class, env, **hyperparameters):
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)                                                   
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

	def learn(self, total_timesteps):  # Learning method 
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration : key role
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			for _ in range(self.n_updates_per_iteration):                                                       
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Backward actor
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Backward critic
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())


	def rollout(self): # Generate trajectory
		# Batch data.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. 
		ep_rews = []
		t = 0 
		
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode
			obs, _ = self.env.reset()
			done = False

			for ep_t in range(self.max_timesteps_per_episode):
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1 
				batch_obs.append(obs)
				action, log_prob = self.get_action(obs)
				obs, rew, terminated, truncated, _ = self.env.step(action)
				done = terminated | truncated
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				
				if done:
					break

			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)


		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
		Compute the Reward-To-Go of each timestep in a batch given the rewards.
		"""
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):
			discounted_reward = 0 
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		return batch_rtgs

	def get_action(self, obs):
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		dist = MultivariateNormal(mean, self.cov_mat)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.99                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA


		if self.seed != None:
			assert(type(self.seed) == int)
			torch.manual_seed(self.seed)

	