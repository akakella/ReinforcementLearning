import gym
import numpy as np
from agents.actor_critic_agent import actor_critic_agent
from gym import wrappers
import math
from matplotlib import pyplot as plt

"""
Trains the cartpole-v0 environment from OpenAI's gym for a predefined number of episodes

Currently uses one-step tabular Q-Learning.
"""

def main(episodes):
	env = gym.make('LunarLander-v2')
	# env = wrappers.Monitor(env, '../tmp/mountaincar-experiment', force=True)

	agent = actor_critic_agent(env, use_kernel=False)

	rewards = agent.train(episodes)
	print('Maximum reward obtained: ' + repr(max(rewards)))
	print('Average reward obtained over last 100 episodes: ' + repr(np.mean(rewards[(episodes-100):])))
	env.close()

if __name__ == "__main__":
	episodes = 3000
	main(episodes)