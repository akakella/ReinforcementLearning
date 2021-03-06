import argparse
import gym
import numpy as np
from temporal_difference.agents.qlearning_agent import qlearning_agent
from temporal_difference.agents.sarsa_agent import sarsa_agent
from function_approximation.agents.semigradient_q_agent import semigradient_q_agent
from function_approximation.agents.semigradient_sarsa_agent import semigradient_sarsa_agent
from policy_gradients.agents.actor_critic_agent import actor_critic_agent
from gym import wrappers
import pandas as pd
import math
from matplotlib import pyplot as plt

"""
Run specified environment with generic agent for a specified number of episodes.
"""

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--environment', required=True)
	parser.add_argument('-a', '--agent', required=True)
	parser.add_argument('-n', '--episodes', required=True)
	args = parser.parse_args()

	environment = args.environment
	agent_type = args.agent
	episodes = args.episodes

	env = gym.make(environment)
	env = wrappers.Monitor(env, '../tmp/' + environment + '-experiment', force=True)

	if agent_type == 'qlearning_agent':
		agent = qlearning_agent(env)
	elif agent_type == 'sarsa_agent':
		agent = sarsa_agent(env)
	elif agent_type == 'semigradient_q_agent':
		agent = semigradient_q_agent(env)
	elif agent_type == 'semigradient_sarsa_agent':
		agent = semigradient_sarsa_agent(env)
	elif agent_type == 'actor_critic_agent':
		agent = actor_critic_agent(env)
	else:
		print('Agent does not exist!')
		return;

	rewards = agent.train(int(episodes))
	print('Maximum reward obtained: ' + repr(max(rewards)))
	reward_frame = pd.DataFrame(rewards)
	rolling_maxes = reward_frame.rolling(window=100, center=False).mean()
	print('Best reward obtained over 100 episodes: ' + repr(np.array(np.max(rolling_maxes))[0]))
	env.close()

if __name__ == "__main__":
	main()