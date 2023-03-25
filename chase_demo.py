import gym
import mujoco_py

count = 0
problem = "Chase-v0"
env = gym.make(problem)

num_states = env.observation_space.shape
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

env.reset()
env.render()
t_total = 10000

for t in range(t_total):
    action = env.action_space.sample()  #随机采样动作
    action[18] = action[18]
    action[22] = action[22]
    observation, reward, done, info = env.step(action)  #与环境交互，获得下一步的时刻
    if done:
        pass
    env.render()
    if t % 100 == 99:
        env.deep_image()
    print("reward:", reward)
    # print("observation:", observation)
    print("out of border:", done)
env.close() 
