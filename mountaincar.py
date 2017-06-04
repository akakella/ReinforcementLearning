import gym
import numpy as np
from FunctionApproximation.semigradient_q_agent import semigradient_q_agent
from FunctionApproximation.semigradient_sarsa_agent import semigradient_sarsa_agent
from PolicyGradients.actor_critic_agent import actor_critic_agent
from gym import wrappers
import math
from matplotlib import pyplot as plt

"""
Trains the cartpole-v0 environment from OpenAI's gym for a predefined number of episodes

Currently uses one-step tabular Q-Learning.
"""

def main(episodes):
	env = gym.make('MountainCar-v0')
	env = wrappers.Monitor(env, '../tmp/mountaincar-experiment', force=True)

	agent = actor_critic_agent(env)

	rewards = agent.train(episodes)
	print('Maximum reward obtained: ' + repr(max(rewards)))
	print('Average reward obtained over last 100 episodes: ' + repr(np.mean(rewards[(episodes-100):])))
	env.close()

if __name__ == "__main__":
	episodes = 1000
	main(episodes)