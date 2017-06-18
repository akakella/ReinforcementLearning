import gym
from gym import wrappers
from agents.dqn_agent import dqn_agent
import math
import numpy as np

"""
Trains the CartPole-v0 environment from OpenAI's gym for a predefined number of episodes

"""

def main(episodes):
	env = gym.make('CartPole-v0')
	#env = wrappers.Monitor(env, '../tmp/cartpole-experiment', force=True)

	agent = dqn_agent(env)
	rewards = agent.train(episodes)


	env.close()

if __name__ == "__main__":
	episodes = 500
	main(episodes)