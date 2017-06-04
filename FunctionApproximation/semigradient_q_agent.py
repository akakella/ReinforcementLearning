import gym
import numpy as np
import math
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.kernel_approximation import RBFSampler

class semigradient_q_agent():
	"""
	Creates a learning agent using Semi-gradient Q-Learning for control. Uses linear function
	approximation with RBF kernels.

	Args:
		env: gym environment for the agent to act on
		agent_params: 
			epsilon_min: minimum allowed epsilon
			decay_rate: decay rate for logarithmic decay of epsilon
			discount: discount rate for future rewards
			iter: maximum number of iterations per episode

	"""

	def __init__(self, env, **agent_params):
		self.env = env

		# Sample feature space and define scaler to detrend data
		observation_samples = np.array([env.observation_space.sample() for x in range (10000)])
		self.detrend = preprocessing.StandardScaler()
		self.detrend.fit(observation_samples)

		# Use detrended data to generate feature space with RBF kernels
		self.featurizer = pipeline.FeatureUnion([
			("rbf1", RBFSampler(gamma=3.0, n_components=100)),
        	("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        	("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        	("rbf4", RBFSampler(gamma=0.5, n_components=100))])
		self.featurizer.fit(self.detrend.transform(observation_samples))

		# Generate linear value function model for each action in our action space
		self.models = []
		initReward = np.array(0)
		for k in range (env.action_space.n):
			self.models.append(linear_model.SGDRegressor(learning_rate="constant"))
			random_features = self.map_to_features(self.env.reset())
			self.models[k].partial_fit(random_features.reshape(1,-1), initReward.ravel())

		self.agent_params = {
			"epsilon_min": 0.01,
			"decay_rate": 0.02,
			"discount": 0.99,
			"iter": 1000
		}
		self.agent_params.update(agent_params)

	def map_to_features(self, state):
		"""
		Maps states to a feature vector

		Args:
			state: observation of the state

		Returns:
			features: feature vector corresponding to state
		"""
		detrended_state = self.detrend.transform(state.reshape(1,-1))
		features = self.featurizer.transform(detrended_state.reshape(1,-1))
		return features[0]

	def get_value(self, state):
		"""
		Gets values for taking all actions in the given state

		Args:
			state: observation of the state

		Returns:
			values: array of values for each action in the action space
		"""
		features = self.map_to_features(state)
		values = []
		for m in self.models:
			values.append(m.predict([features])[0])

		return values

	def act(self, state, epsilon):
		"""
		Determine action based on state

		Args:
			state: observation of the state

		Returns:
			action: action to take according to epsilon-greedy policy
		"""
		if (np.random.random() > epsilon):
			values = self.get_value(state)
			action = np.argmax(values)
		else:
			action = self.env.action_space.sample()

		return action

	def train(self, episodes):
		"""
		Train agent for a given number of episodes

		Args:
			episodes: number of episodes to run

		Returns:
			rewards: array of rewards obtained by episode
		"""

		rewards = []
		for ep in range(episodes):
			state = self.env.reset()
			epsilon = max(self.agent_params["epsilon_min"], min(1, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))
			epReward = 0

			for t in range(self.agent_params["iter"]):
				action = self.act(state, epsilon)
				next_state, reward, done, _ = self.env.step(action)
				
				values = self.get_value(next_state)

				reward_target = np.array(reward + self.agent_params["discount"]*values[np.argmax(values)])
				state_features = self.map_to_features(state)
				self.models[action].partial_fit(state_features.reshape(1,-1), reward_target.ravel())
				
				state = next_state

				epReward += reward
				if done:
					rewards.append(epReward)
					break

		return rewards