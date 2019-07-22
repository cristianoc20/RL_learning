import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0') #定义使用gym库中的哪一个环境
env = env.unwrapped #还原env的原始设置，env外包了一层防作弊层

print(env.action_space) #查看这个环境可用的action有多少个
print(env.observation_space) #查看这个环境中可用的state的observation有多少个
print(env.observation_space.high) #查看observation最高取值
print(env.observation_space.low) #查看observation最低取值

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):
    #获取回合i_episode第一个observation
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()#刷新环境

        action = RL.choose_action(observation)#选行为

        observation_, reward, done, info = env.step(action)#获取下一个state

        # x是车的水平位移，所以r1是车越偏离中心，得分（reward）越少
        # theta是棒子离垂直的角度，角度越大，越不垂直。所以r2是棒越垂直，分(reward)越高
        # 总reward是r1和r2的结合，既考虑位置也考虑角度，这样DQN学习更有效率
        x, x_dot, theta, theta_dot = observation_

        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        #保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()