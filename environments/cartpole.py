import gym
from q_agent import QAgent
from gym import wrappers

def main(episodes):
	env = gym.make("CartPole-v0")
	env = wrappers.Monitor(env, '../tmp/cartpole-experiment', force=True)
	action_space = [0, 1]
	observation_min = [-3, -2, -5, -3]
	observation_max = [3, 2, 5, 3]
	num_bins = 100

	agent = QAgent(action_space, observation_min, observation_max, num_bins)

	for i_episode in range(episodes):
		obs = env.reset()
		agent.train(env)
	
if __name__ == "__main__":
	episodes = 400
	main(episodes)