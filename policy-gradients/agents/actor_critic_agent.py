import gym
import numpy as np
import math
import tensorflow as tf
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.kernel_approximation import RBFSampler

class actor_critic_agent():
	"""
	Creates a learning agent using the Actor-Critic algorithm with linear function approximation
	and RBF kernels.
	
	Args:
		env: gym environment for the agent to act on
		agent_params: 
			discount: discount rate for future rewards
			iter: maximum number of iterations per episode
	"""
	
	def __init__(self, env, **agent_params):
		self.env = env
		self.agent_params = {
			"discount": 0.99,
			"iter": 1000,
			"epsilon": 0.01
		}
		self.agent_params.update(agent_params)
		# Preprocessor for detrending observations
		observation_samples = np.array([env.observation_space.sample() for x in range (10000)])
		self.detrend = preprocessing.StandardScaler()
		self.detrend.fit(observation_samples)

		# Generating feature space of RBF kernels
		self.featurizer = pipeline.FeatureUnion([
			("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        	("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        	("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        	("rbf4", RBFSampler(gamma=0.5, n_components=100))])
		self.featurizer.fit(self.detrend.transform(observation_samples))

		# Generating linear model approximation for value function
		with tf.variable_scope("value_function"):
			self.value_features = tf.placeholder(tf.float32, [400], name="value_features")
			self.value_reward_target = tf.placeholder(tf.float32, name="value_reward_target")
			value_output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.value_features, 0), 
				num_outputs=1, activation_fn=None, weights_initializer=tf.zeros_initializer)
			
			self.value_estimate = tf.squeeze(value_output_layer)
			self.value_loss = tf.squared_difference(self.value_estimate, self.value_reward_target)
			self.value_optimizer = tf.train.AdamOptimizer()
			self.value_train_op = self.value_optimizer.minimize(self.value_loss)    

		# Generating linear model approximation for policy function
		with tf.variable_scope("policy_function"):
			self.action = tf.placeholder(tf.int32, name="action")
			self.policy_features = tf.placeholder(tf.float32, [400], name="policy_features")
			self.policy_reward_target = tf.placeholder(tf.float32, name="policy_reward_target")
			policy_output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.policy_features, 0),
				num_outputs=env.action_space.n, activation_fn=None, weights_initializer=tf.zeros_initializer)
			
			self.action_probabilities = tf.squeeze(tf.nn.softmax(policy_output_layer))
			self.max_action_probability = tf.gather(self.action_probabilities, self.action)

			self.policy_loss = -tf.log(self.max_action_probability)*self.policy_reward_target
			self.policy_optimizer = tf.train.AdamOptimizer()
			self.policy_train_op = self.policy_optimizer.minimize(self.policy_loss)


	def choose_action(self, state):
		"""
		Chooses action based on a given observation
		
		Args:
			state: observation returned from the environment 
			
		Returns:
			action: action based on epsilon-greedy policy function
		"""
		if (np.random.random() > self.agent_params["epsilon"]):
			sess = tf.get_default_session()
			features = self.map_to_features(state)
			action_probabilities = sess.run(self.action_probabilities, {self.policy_features: features})
			return np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
		else:
			return self.env.action_space.sample()


	def map_to_features(self, state):
		"""
		Maps state of environment to features

		Args:
			state: state of the environment

		Returns:
			features: feature mapping for given state
		"""
		detrended_state = self.detrend.transform(state.reshape(1,-1))
		features = self.featurizer.transform(detrended_state.reshape(1,-1))
		return features[0]

	def predict_value(self, state):
		"""
		Predicts value of state using the value function

		Args:
			state: current state of the system

		Returns:
			predicted_value: value returned from the value function
		"""
		sess = tf.get_default_session()
		features = self.map_to_features(state)
		value = sess.run(self.value_estimate, {self.value_features: features})
		return value


	def train(self, episodes):
		"""
		Runs the agent for a set number of episodes and updates the value table
		
		Args:
			episodes: number of episodes to train over

		Returns:
			rewards: array of rewards obtained by episode
			
		"""
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			rewards = []

			for ep in range(episodes):
				state = self.env.reset()

				epReward = 0
				for i in range(self.agent_params["iter"]):
					action = self.choose_action(state)

					next_state, reward, done, info = self.env.step(action)

					reward_target = reward + self.agent_params["discount"]*self.predict_value(next_state) - self.predict_value(state)

					_, value_loss = sess.run([self.value_train_op, self.value_loss], {self.value_features: self.map_to_features(state), 
						self.value_reward_target: reward_target})
					_, policy_loss = sess.run([self.policy_train_op, self.policy_loss], {self.policy_features: self.map_to_features(state), 
						self.policy_reward_target: reward_target, self.action: action})
					

					epReward += reward
					state = next_state
					if done:
						rewards.append(epReward)
						break

		return rewards



