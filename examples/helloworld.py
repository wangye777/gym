import gym
env = gym.make('ObjectTransition-v0')
#env = gym.make('MountainCarContinuous-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
	print("action={}, observation={}, reward={}, done={}".format(action, observation, reward, done))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
