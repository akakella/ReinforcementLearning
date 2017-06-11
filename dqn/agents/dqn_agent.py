import gym
import numpy as np
import random
import math
import tensorflow as tf

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
            self.state = tf.placeholder(shape=[None, np.size(env.observation_space.sample())], dtype=tf.float32)
            self.reward_target = tf.placeholder(shape=[None, np.size(env.action_space.sample())], dtype=tf.float32)

            input_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                            num_outputs = layers[1],
                                                            activation_fn = tf.nn.relu,
                                                            weights_initializer=tf.zeros_initializer)
            hidden_layer = []
            layer = tf.contrib.layers.fully_connected(inputs = input_layer,
                                                               num_outputs = layers[1],
                                                               activation_fn = tf.nn.relu,
                                                               weights_initializer=tf.zeros_initializer)
            hidden_layer.append(layer)
            for k in range(layers[0]):
                layer = tf.contrib.layers.fully_connected(inputs = hidden_layer[k],
                                                                     num_outputs = layers[1],
                                                                     activation_fn = tf.nn.relu,
                                                                     weights_initializer=tf.zeros_initializer)
                hidden_layer.append(layer)

            value_output_layer = tf.contrib.layers.fully_connected(inputs=hidden_layer[layers[0]],
                                                                   num_outputs = env.action_space.n,
                                                                   activation_fn = None,
                                                                   weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(value_output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.reward_target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_opt = self.optimizer.minimize(self.loss)

    def predict(self, sess, state):
        """
        Predict the values for a given state

        Args:
            state: state of interest

        Return:
            q_values: array of values for each action

        """
        q_values = sess.run(self.value_estimate, {self.state: state})
        return q_values

    def update(self, sess, state, reward_target):
        """
        Updates the model with the latest batch of state/reward pairs

        Args:
            state: batch of states
            reward_target: batch of reward targets corresponding to states

        Returns:
            loss: loss for update setp
        """
        _, loss = sess.run([self.train_opt, self.loss], {self.state: state, self.reward_target: reward_target})
        return loss


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

    def __init__(self, env, layers=[10,100], batch_size = 128, replay_memory_size=100000, **agent_params):
        self.env = env
        
        self.estimator = estimator(env, "estimator", layers)
        self.target_estimator = estimator(env, "target", layers)
        self.target_update_frequency = 1000

        self.agent_params = {
            "epsilon_min": 0.001,
            "decay_rate": 0.02,
            "discount": 0.99,
            "iter": 200,
        }
        self.agent_params.update(agent_params)
        self.replay_memory = []
        
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size


    def act(self, sess, state, epsilon):
        """
        Choose an action based on an epsilon-greedy policy and the Q-value estimator

        Args:
            sess: current tensorflow session
            state: where the agent currently is
            epsilon: curent epsilon

        Returns:
            action: action according to epsilon-greedy policy
        """

        if (np.random.random() > epsilon):
            action_values = self.estimator.predict(sess, state.reshape(1,-1))
            action = np.argmax(action_values)
        else:
            action = self.env.action_space.sample()

        return action


    def update_target_estimator(self, sess):
        """
        Update the target estimator's weights with those from the true estimator
        
        Args:
            sess: current tensorflow session
        """

        estimator_params = []
        target_params = []
        for v in tf.trainable_variables():
            if (v.name.startswith(self.estimator.scope)):
                estimator_params.append(v)
            elif (v.name.startswith(self.target_estimator.scope)):
                target_params.append(v)

        target_params = sorted(target_params, key = lambda x: x.name)
        estimator_params = sorted(estimator_params, key = lambda x: x.name)

        updates = []
        for tvar, evar in zip(target_params, estimator_params):
            assignment = tvar.assign(evar)
            updates.append(assignment)

        sess.run(updates)


    def init_replay_memory(self):
        """ 
        Initialize the replay memory with random (S,A,R,S') transitions
        """

        state = self.env.reset()
        for m in range(self.replay_memory_size):
            action = self.env.action_space.sample()

            next_state, reward, done, _ = self.env.step(action)
            self.update_replay_memory((state, action, reward, next_state))

            if done:
                state = env.reset()
            else:
                state = next_state

    def update_replay_memory(self, transition):
        """
        Update replay memory with the latest transition

        Args:
            transition: latest transition from the agent
        """

        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        self.replay_memory.append(transition)

    def train(self, episodes):
        """
        Trains DQN
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if not self.replay_memory:
            self.init_replay_memory()

        rewards = []
        timestep = 0
        for ep in range(episodes):
            state = self.env.reset()
            epReward = 0
            epsilon = max(self.agent_params["epsilon_min"], min(1, 1.0 - math.log10((ep + 1)*self.agent_params["decay_rate"])))

            for t in range(500):
                action = self.act(sess, state, epsilon)

                next_state, reward, done, _ = self.env.step(action)

                self.update_replay_memory((state, action, reward, next_state))

                ### Form batch states and targets
                samples = random.sample(self.replay_memory, self.batch_size)
                batch_states = np.zeros((self.batch_size, 4))
                batch_rewards = np.zeros((self.batch_size, 1))
                for t_idx, transitions in enumerate(samples):
                    batch_states[t_idx] = transitions[0]
                    reward_target = self.estimator.predict(sess, transitions[0].reshape(1,-1))
                    reward_target[transitions[1]] = transitions[2] + self.agent_params["discount"]*np.max(self.target_estimator.predict(sess, transitions[3].reshape(1,-1)))
                    batch_rewards[t_idx] = reward_target

                self.estimator.update(sess, batch_states, batch_rewards.reshape(-1,1))

                if (timestep % self.target_update_frequency) == 0:
                    self.update_target_estimator(sess)

                epReward += reward
                state = next_state
                timestep += 1
                if done:
                    rewards.append(epReward)
                    break

            print("Episode: {}, Reward: {}".format(ep, epReward))

        return rewards