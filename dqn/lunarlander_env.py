import gym
from gym import wrappers
from agents.dqn_agent import dqn_agent
import math
import numpy as np
import pandas as pd
import math

"""
Trains the LunarLander-v2 environment from OpenAI's gym for a predefined number of episodes

"""

def main(episodes):
	# Init replay memory
	env = gym.make('LunarLander-v2')
	replay_agent = dqn_agent(env, layers=[64,256,64], decay_rate=0.02, batch_size=64, target_update_frequency=100)
	replay_agent.init_replay_memory()

	train_env = gym.make('LunarLander-v2')
	train_env = wrappers.Monitor(train_env, '../tmp/lunarlander-experiment', force=True)
	agent = dqn_agent(train_env, layers=[64,256,64], decay_rate=0.02, batch_size=64, target_update_frequency=100)
	agent.replay_memory = replay_agent.replay_memory
	rewards = agent.train(episodes)


	print('Maximum reward obtained: ' + repr(max(rewards)))
	reward_frame = pd.DataFrame(rewards)
	rolling_maxes = reward_frame.rolling(window=100, center=False).mean()
	print('Best reward obtained over 100 episodes: ' + repr(np.array(np.max(rolling_maxes))[0]))
	env.close()
	train_env.close()

if __name__ == "__main__":
	episodes = 1000
	main(episodes)