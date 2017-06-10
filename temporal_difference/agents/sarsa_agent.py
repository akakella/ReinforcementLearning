import gym
import numpy as np
import math
from collections import defaultdict

class sarsa_agent():
	"""
	Creates a learning agent using the SARSA algorithm. Currently does not work for continuous
	action spaces.
	
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
	
	def __init__(self, env, n_actions=None, obs_min=None, obs_max=None, num_bins=None, **agent_params):
		self.env = env

		self.n_actions = env.action_space.n if (n_actions == None) else n_actions
		self.discrete_obs_space = False
		if type(env.observation_space) is gym.spaces.discrete.Discrete:
			self.discrete_obs_space = True
		else:
			self.obs_min = env.observation_space.low if (obs_min == None) else obs_min
			self.obs_max = env.observation_space.high if (obs_max == None) else obs_max
			self.num_bins = [10] * len(env.observation_space.sample()) if (num_bins == None) else num_bins
		
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

	def discretize_state(self, observation):
		"""
		Converts true observation to binned observation
		
		Args:
			observation: observation array returned from the environment
			
		Returns:
			indices: tuple of indices for bins the observation corresponds to
		"""
		indices = []
		if self.discrete_obs_space:
			return observation
		else:
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
		Runs the agent for a set number of episodes and updates the value table
		
		Args:
			episodes: number of episodes to train over

		Returns:
			rewards: array of rewards obtained by episode
			
		"""
		
		rewards = []
		for ep in range(episodes):
			obs = self.env.reset()
			obs = self.discretize_state(obs)
			epsilon = max(self.agent_params["epsilon_min"], min(1, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))
			alpha = max(self.agent_params["alpha_min"], min(0.5, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))

			action = self.act(obs, epsilon)
			epReward = 0
			for t in range(self.agent_params["iter"]):
				next_obs, reward, done, info = self.env.step(action)
				next_obs = self.discretize_state(next_obs)

				next_act = self.act(next_obs, epsilon)

				self.qtable[obs][action] = self.qtable[obs][action] + alpha*(reward + self.agent_params["discount"]*self.qtable[next_obs][next_act] - self.qtable[obs][action])
				obs = next_obs
				action = next_act
				epReward += reward

				if done:
					rewards.append(epReward)
					break
		return rewards
