import gym
import numpy as np
import math
from collections import defaultdict

class SarsaAgent():
	"""
	Creates a learning agent using the SARSA algorithm
	
	Args:
		action_space: list of available actions for the agent
		obs_min: minimum possible values for each of the observations
		obs_max: maximum possible values for each of the observations
		num_bins: number of bins to create for each observation in the discretized observation space
		agent_params: array of agent parameters detailed below
			mean: initial mean reward value for each state-action pair
			std: standard deviation for initial mean reward value for each state-action pair
			alpha_min: minimum learning rate
			epsilon_min: minimum probability to select a random action
			decay_rate: decay rate for logarithmic decay of epsilon and alpha
			discount: scale factor for discounting future state rewards
			iter: maximum number of iterations per episode
	"""
	
	def __init__(self, env, n_actions, obs_min, obs_max, num_bins, **agent_params):
		self.n_actions = n_actions
		self.obs_min = obs_min
		self.obs_max = obs_max
		self.num_bins = num_bins
		self.env = env
		
		self.agent_params = {
			"mean" : 0,
			"std" : 0,
			"alpha_min" : 0.1,
			"epsilon_min" : 0.01,
			"decay_rate" : 0.04,
			"discount" : 0.99,
			"iter" : 250
		}

		self.agent_params.update(agent_params)
		self.qtable = defaultdict(lambda: self.agent_params["std"] * np.random.randn(self.n_actions) + self.agent_params["mean"])

	def convertToObsSpace(self, observation):
		"""
		Converts true observation to binned observation
		
		Args:
			observation: observation array returned from the environment
			
		Returns:
			indices: tuple of indices for bins the observation corresponds to
		"""
		indices = []

		for i in range(len(observation)):
			if observation[i] <= self.obs_min[i]:
				idx = 0
			elif observation[i] >= self.obs_max[i]:
				idx = self.num_bins[i] - 1
			else:
				offset = (self.num_bins[i]-1)*self.obs_min[i]/(self.obs_max[i] - self.obs_min[i])
				scale = (self.num_bins[i]-1)/(self.obs_max[i] - self.obs_min[i]) 
				idx = int(round(scale*observation[i] - offset))
			indices.append(idx)
		return tuple(indices)

	def act(self, observation, epsilon):
		"""
		Chooses action based on a given observation
		
		Args:
			observation: observation returned from the environment (post-scalar conversion)
			epsilon: epsilon to use for epsilon-greedy algorithm
			
		Returns:
			action: action based on epsilon-greedy algorithm
		"""
		if (np.random.random() > epsilon):
			action = np.argmax(self.qtable[observation])
		else:
			action = self.env.action_space.sample()

		return action

	def train(self, episodes):
		"""
		Resets environment and runs each episode
		
		Args:
			episodes: number of episodes to train over

		Returns:
			maxReward: maximum reward obtained over this training
			
		"""
		maxReward = -float('inf')
		totalReward = -float('inf')
		for ep in range(episodes):
			if (maxReward < totalReward):
				maxReward = totalReward
			totalReward = 0

			obs = self.env.reset()
			obs = self.convertToObsSpace(obs)
			epsilon = max(self.agent_params["epsilon_min"], min(1, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))
			alpha = max(self.agent_params["alpha_min"], min(0.5, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))

			action = self.act(obs, epsilon)
			for t in range(self.agent_params["iter"]):
				next_obs, reward, done, info = self.env.step(action)
				next_obs = self.convertToObsSpace(next_obs)

				next_act = self.act(next_obs, epsilon)

				self.qtable[obs][action] = self.qtable[obs][action] + alpha*(reward + self.agent_params["discount"]*self.qtable[next_obs][next_act] - self.qtable[obs][action])
				obs = next_obs
				action = next_act
				totalReward += reward

				if done:
					break
		return maxReward
