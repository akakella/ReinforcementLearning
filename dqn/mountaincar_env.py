import gym
from gym import wrappers
from agents.dqn_agent import dqn_agent
import math
import numpy as np

"""
Trains the MountainCar-v0 environment from OpenAI's gym for a predefined number of episodes

"""

def main(episodes):
	env = gym.make('LunarLander-v2')
	env = wrappers.Monitor(env, '../tmp/cartpole-experiment', force=True)

	agent = dqn_agent(env, layers=[2,128], decay_rate=0.02)
	rewards = agent.train(episodes)


	print('Maximum reward obtained: ' + repr(max(rewards)))
	reward_frame = pd.DataFrame(rewards)
	rolling_maxes = reward_frame.rolling(window=100, center=False).mean()
	print('Best reward obtained over 100 episodes: ' + repr(np.array(np.max(rolling_maxes))[0]))
	env.close()

if __name__ == "__main__":
	episodes = 2000
	main(episodes)