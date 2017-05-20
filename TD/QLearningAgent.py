import gym
import numpy as np
from collections import defaultdict

class QLearningAgent():
	"""
	Creates a learning agent using tabular Q-Learning
	
	Args:
		action_space: list of available actions for the agent
		obs_min: minimum possible values for each of the observations
		obs_max: maximum possible values for each of the observations
		num_bins: number of bins to create for each observation in the discretized observation space
		agent_params: array of agent parameters detailed below
			mean: initial mean reward value for each state-action pair
			std: standard deviation for initial mean reward value for each state-action pair
			learning_rate: scale factor for incorporating new information
			eps: probability to select a random action
			discount: scale factor for discounting future state rewards
			iter: maximum number of iterations per episode
	"""
	
	def __init__(self, action_space, obs_min, obs_max, num_bins, **agent_params):
		self.action_space = action_space
		self.n_actions = np.size(action_space)
		self.obs_bins = []
		self.num_bins = num_bins

		for i, item in enumerate(obs_min):
			bin = np.arange(obs_min[i], obs_max[i], (float(obs_max[i]) - float(obs_min[i]))/(self.num_bins-1))
			self.obs_bins.append(bin)
		
		self.agent_params = {
			"mean" : 0,
			"std" : 0,
			"learning_rate" : 0.1,
			"eps" : 0.02,
			"discount" : 0.95,
			"iter" : 1000
		}

		self.agent_params.update(agent_params)
		self.qtable = defaultdict(lambda: self.agent_params["std"] * np.random.randn(self.n_actions) + self.agent_params["mean"])

	def convertToObsSpace(self, observation):
		"""
		Converts observation array to scalar value for easier indexing in the value table
		
		Args:
			observation: observation array returned from the environment
			
		Returns:
			obs: scalar value corresponding to the input observation
		"""
		obs = 0
		
		for i, item in enumerate(observation.flatten()):
			obs = obs + np.digitize(item, self.obs_bins[i]) * pow(pow(10, np.ceil(np.log10(self.num_bins))), i)
			
		return obs

	def act(self, observation):
		"""
		Chooses action based on a given observation
		
		Args:
			observation: observation returned from the environment (post-scalar conversion)
			
		Returns:
			action: action based on epsilon-greedy algorithm
		"""
		if (np.random.random() > self.agent_params["eps"]):
			action = np.argmax(self.qtable[observation])
		else:
			action = np.random.choice(self.action_space)

		return action

	def train(self, env):
		"""
		Resets environment and runs each episode
		
		Args:
			env: desired environment for agent to act on
			
		"""
		obs = env.reset()
		obs = self.convertToObsSpace(obs)

		for t in range(self.agent_params["iter"]):
			action = self.act(obs)
			next_obs, reward, done, info = env.step(action)
			next_obs = self.convertToObsSpace(next_obs)
			if done:
				future = 0
				break;
			else:
				future = np.max(self.qtable[next_obs])

			self.qtable[obs][action] = self.qtable[obs][action] + self.agent_params["learning_rate"] * (reward + self.agent_params["discount"]*future - self.qtable[obs][action])
			obs = next_obs