import argparse
import gym
import numpy as np
from TD.qlearning_agent import qlearning_agent
from TD.sarsa_agent import sarsa_agent
from FunctionApproximation.semigradient_q_agent import semigradient_q_agent
from FunctionApproximation.semigradient_sarsa_agent import semigradient_sarsa_agent
from PolicyGradients.actor_critic_agent import actor_critic_agent
from gym import wrappers
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
	print('Average reward obtained over last 100 episodes: ' + repr(np.mean(rewards[(int(episodes)-100):])))
	env.close()

if __name__ == "__main__":
	main()