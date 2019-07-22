# 一、前言
手动编写环境是一件很耗时间的事情，所以如果可以直接使用比人编写好的环境，可以节约我们很多时间。OpenAI gym就是这样一个模块，他提供给我们很多优秀的模拟环境。我们的各种强化学习算法都能使用这些环境。之前的环境都是用``tkinter``来手动编写，或者想玩玩更厉害的，像OpenAI一样，使用pyglet模块来编写。
OpenAI gym官网：[https://gym.openai.com/](https://gym.openai.com/)

我们可以先看看OpenAI gym有哪些游戏：
有2D的：
![](http://puiarp73w.bkt.clouddn.com/Fq8TnOgRA1sPhi8mCOXpWn2PrAqQ)
也有3D的：
![](http://puiarp73w.bkt.clouddn.com/Fp-Z2Qk0MWctgfpsmelcWCR4ulYh)
本次将会以CartPole和MountainCar两个经典例子来给大家说明。

# 二、安装
笔者电脑是Ubuntu16.04，可以直接复制下面代码安装：
```python
# python 2.7, 复制下面
$ pip install gym

# python 3.5, 复制下面
$ pip3 install gym
```
如果没有报错那就安装好gym(基本款)，可以玩以下游戏：
- ``algorithmic``
- ``toy_text``
- ``classic_control``(这个需要pyglet模块)

如果你想玩gym提供的全套游戏，则使用以下代码：
```python
# python 2.7, 复制下面
$ pip install gym[all]

# python 3.5, 复制下面
$ pip3 install gym[all]
```
# 三、CartPole例子
这个游戏的目的是让小车尽量不偏离中心以及棍子尽量垂直，我们可以看下面的示例图，经过训练后小车就会尽量呆在中间棍子也基本保持垂直。
![](http://puiarp73w.bkt.clouddn.com/FqjUy3CFsZP51ebaxEpDDWIv26RD)
## 主循环
我们还是采用DQN的方式来实现RL，完整代码最后会给我的github链接。
```python
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
```
这是更为典型的RL cost曲线：
![](http://puiarp73w.bkt.clouddn.com/Fg6T8M2-P7t2yhaiiTPrrHKAmuM5)
# 四、MountainCar例子
小车经过谷底的震荡，慢慢地就可以爬到山顶拿旗子了。
![](http://puiarp73w.bkt.clouddn.com/FlTXAHa6JQZZ7AaPIRbRstw9DS-X)
代码和上面差不多，只是定义的reward不同：
```python
import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.001, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.0002,)

total_steps = 0


for i_episode in range(10):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # 车开的越高reward越大
        reward = abs(position - (-0.5))     # r in [0, 1]

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
```
出来的cost曲线是这样的：
![](http://puiarp73w.bkt.clouddn.com/Fl-2g3j3qFUa1MrMBS2vBk_sMqyh)

完整代码放在我的github上面：[https://github.com/cristianoc20/RL_learning](https://github.com/cristianoc20/RL_learning)

参考：[https://github.com/MorvanZhou](https://github.com/MorvanZhou)
