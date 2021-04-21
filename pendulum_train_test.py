import gym
import numpy as np
env = gym.make('CartPole-v1')
env.seed(42)

# basic control policy
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for i_episode in range(500):
    episode_rewards = 0
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = basic_policy(observation) # create action based on policy
        observation, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
env.close()
