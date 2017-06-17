import gym
import numpy as np
import random
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class estimator():
    """
    Creates an Q-value estimation network

    Args:
        scope: scope to assign to network parameters
        layers: contains number of layers and number of hidden units
    """

    def __init__(self, env, scope, layers):
        self.scope = scope
        with tf.variable_scope(scope):
            self.state_dim = env.observation_space.shape[0]
            self.n_actions = env.action_space.n

            self.model = Sequential()
            self.model.add(Dense(units=layers[1], activation='relu', input_dim=self.state_dim))

            hidden_layer = []
            for k in range(layers[0]):
                self.model.add(Dense(units=layers[1], activation='relu'))

            self.model.add(Dense(units=self.n_actions, activation='linear'))
            train_opt = RMSprop(lr=0.00025)
            self.model.compile(loss='mean_squared_error', optimizer=train_opt)

    def predict(self, state, batch=False):
        """
        Predict the values for a given state

        Args:
            state: state of interest

        Return:
            q_values: array of values for each action

        """
        if batch:
            q_values = self.model.predict(state)
        else:
            q_values = self.model.predict(state.reshape(1, self.state_dim)).flatten()
        
        return q_values

    def update(self, state, reward_target):
        """
        Updates the model with the latest batch of state/reward pairs

        Args:
            state: batch of states
            reward_target: batch of reward targets corresponding to states

        """
        self.model.fit(state, reward_target, epochs=1, verbose=0)
        
    def set_weights(self, weights):
        """
        Transfers weights from given estimator to current model
        
        Args:
            weights: weights to apply to new model
        """
        self.model.set_weights(weights)
        
    def get_weights(self):
        """
        Get the weights from model inside this estimator
        
        Returns:
            weights: weights from this model
        """
        return self.model.get_weights()


class dqn_agent():
    """
    Creates a learning agent using Deep Q-Learning with memory replay and target networks. Only
    works for discrete action spaces.

    Args:
        env: gym environment for the agent to act on
        agent_params: 
            epsilon_min: minimum allowed epsilon
            decay_rate: decay rate for logarithmic decay of epsilon
            discount: discount rate for future rewards
            iter: maximum number of iterations per episode

    """

    def __init__(self, env, layers=[0,64], batch_size = 64, replay_memory_size=100000, **agent_params):
        self.env = env
        
        self.estimator = estimator(env, "estimator", layers)
        self.target_estimator = estimator(env, "target", layers)
        self.target_update_frequency = 1000

        self.agent_params = {
            "epsilon_min": 0.01,
            "decay_rate": 0.04,
            "discount": 0.99,
            "iter": 200,
        }
        self.agent_params.update(agent_params)
        self.replay_memory = []
        
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size


    def act(self, state, epsilon):
        """
        Choose an action based on an epsilon-greedy policy and the Q-value estimator

        Args:
            sess: current tensorflow session
            state: where the agent currently is
            epsilon: curent epsilon

        Returns:
            action: action according to epsilon-greedy policy
        """

        if (random.random() > epsilon):
            action_values = self.estimator.predict(state)
            action = np.argmax(action_values)
        else:
            action = self.env.action_space.sample()
        return action


    def update_target_estimator(self):
        """
        Update the target estimator's weights with those from the true estimator
        
        Args:
            sess: current tensorflow session
        """
        self.target_estimator.set_weights(self.estimator.get_weights())


    def init_replay_memory(self):
        """ 
        Initialize the replay memory with random (S,A,R,S') transitions
        """

        state = self.env.reset()
        for m in range(self.replay_memory_size):
            action = self.env.action_space.sample()

            next_state, reward, done, _ = self.env.step(action)

            if done:
                next_state = None

            self.update_replay_memory((state, action, reward, next_state))

            if done:
                state = self.env.reset()
            else:
                state = next_state

    def update_replay_memory(self, transition):
        """
        Update replay memory with the latest transition

        Args:
            transition: latest transition from the agent
        """

        if len(self.replay_memory) >= self.replay_memory_size:
            self.replay_memory.pop(0)

        self.replay_memory.append(transition)

    def batch_update(self):
        """
        Randomly sample from replay memory and update network.
        
        """
        samples = random.sample(self.replay_memory, self.batch_size)
        batch_states = np.zeros((self.batch_size, np.size(self.env.observation_space.sample())))
        batch_rewards = np.zeros((self.batch_size, self.env.action_space.n))

        # For "None" states, replace them with arbitrary states to allow for batch prediction
        estimated_values = self.target_estimator.predict(np.array([(self.env.observation_space.sample() if t[3] is None else t[3]) for t in samples]), batch=True)
        reward_targets = self.estimator.predict(np.array([t[0] for t in samples]), batch=True)
        for t_idx, transitions in enumerate(samples):
            batch_states[t_idx] = transitions[0]
            reward_target = reward_targets[t_idx]

            if transitions[3] is not None:
                reward_target[transitions[1]] = transitions[2] + self.agent_params["discount"]*np.max(estimated_values[t_idx])
            else:
                reward_target[transitions[1]] = transitions[2]

            batch_rewards[t_idx] = reward_target

        self.estimator.update(batch_states, batch_rewards)

    def train(self, episodes):
        """
        Trains agent for a number of episodes.

        Args:
            episodes: number of episodes to train over
        """

        if not self.replay_memory:
            self.init_replay_memory()

        rewards = []
        timestep = 0
        for ep in range(episodes):
            state = self.env.reset()
            epReward = 0
            epsilon = 0.01 + (1 - 0.01) * math.exp(-0.001 * timestep)

            while True:
                action = self.act(state, epsilon)

                next_state, reward, done, _ = self.env.step(action)

                if done:
                    next_state = None

                self.update_replay_memory((state, action, reward, next_state))
                
                self.batch_update()

                if (timestep % self.target_update_frequency) == 0:
                    self.update_target_estimator()

                epReward += reward
                state = next_state
                timestep += 1
                if done:
                    rewards.append(epReward)
                    break

            print("Episode: {}, Reward: {}".format(ep, epReward))

        return rewards