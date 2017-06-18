import gym
from gym import wrappers
from agents.dqn_agent import dqn_agent
import math
import numpy as np
import pandas as pd

"""
Trains the MountainCar-v0 environment from OpenAI's gym for a predefined number of episodes

"""

def main(episodes):
	# Init replay memory
	env = gym.make('MountainCar-v0')
	replay_agent = dqn_agent(env, layers=[128,256,128], decay_rate=0.02, batch_size=64, target_update_frequency=200)
	replay_agent.init_replay_memory()

	train_env = gym.make('MountainCar-v0')
	train_env = wrappers.Monitor(train_env, '../tmp/mountaincar-experiment', force=True)
	agent = dqn_agent(train_env, layers=[128,256,128], decay_rate=0.02, batch_size=64, target_update_frequency=200)
	agent.replay_memory = replay_agent.replay_memory
	rewards = agent.train(episodes)


	print('Maximum reward obtained: ' + repr(max(rewards)))
	env.close()
	train_env.close()

	reward_frame = pd.DataFrame(rewards)
	rolling_maxes = reward_frame.rolling(window=100, center=False).mean()
	print('Best reward obtained over 100 episodes: ' + repr(np.array(np.max(rolling_maxes))[0]))

if __name__ == "__main__":
	episodes = 1000
	main(episodes)