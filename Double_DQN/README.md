# 1.前言
本篇教程是基于Deep Q network(DQN)的教程，缩减了在DQN方面的介绍，着重强调Double DQN和DQN的不同之处。

接下来我们说说为什么会有Double DQN这种算法，所以我们从Double DQN相对于Natural DQN（传统DQN）的优势说起。

一句话概括，DQN基于Q-Learning，Q-Learning中有``Qmax``，``Qmax``会导致``Q现实``当中的过估计(overestimate)。而Double DQN就是用来解决出现的过估计问题的。在实际问题中，如果你输出你的DQN的Q值，可能就会发现，Q值都超级大，这就是出现了overestimate。

这次的Double DQN的算法实战基于的是OpenAI Gym中的``Pendulum``环境。以下是本次实战结果，目的是经过训练保持杆子始终向上：
![](http://puiarp73w.bkt.clouddn.com/Fg8NwtdV4l0_TLHBDn-1h_nDaLFR)

# 2.算法
我们知道DQN的神经网络部分可以看成一个``最新的神经网络``+``老神经网络``,他们有相同的结构，但内部的参数更新却有时差（TD差分，老神经网络的参数是隔一段时间更新），而它的``Q现实``部分是这样的：
![](http://puiarp73w.bkt.clouddn.com/FhYC0bln28Vo8X_PdP6zQ-zmRtOl)
因为我们的神经网络预测``Qmax``本来就有误差，而每次更新也是向着最大误差的``Q现实``改进神经网络，就是因为这个``Qmax``导致了overestimate。所以Double DQN的想法就是引入另一个神经网络来打消一些最大误差的影响。而DQN中本来就有两个神经网络，所以我们就可以利用一下DQN这个地理优势。我们使用``Q估计``的神经网络估计``Q现实``中`` Qmax(s', a')``的最大动作值。然后用这个被``Q估计``初级出来的动作来选择``Q现实``中的``Q(s')``。总结一下：

有两个神经网络：``Q_eval``（Q估计中的），``Q_next``(Q现实中的)。

原本的``Q_next = max(Q_next(s', a_all))``

而现在Double DQN 中的``Q_next = Q_next(s', argmax(Q_eval(s', a_all)))``，也可以表达成下面那样
![](http://puiarp73w.bkt.clouddn.com/Fi5vUGh_R5LOJY1sCQUsndWevoVX)
## 2.1更新方法
这里的代码都是基于之前的DQN中的代码，在``RL_brain``中，我们将class的名字改成``DoubleDQN``，为了对比Natural DQN，我们也保留原来大部分的DQN的代码。我们在_init_中加入一个double_q参数来表示使用的是Natural DQn还是Double DQN，为了对比的需要，我们的``tf.Session()``也单独传入，并移除原本在 DQN 代码中的这一句:``self.sess.run(tf.global_variables_initializer())``

我们对比Double DQN和Natural DQN在tensorboard中的图，发现他们的结构并没有不同，但是在计算``q_target(也就是Q现实)``的时候，方法是不同的。
![](http://puiarp73w.bkt.clouddn.com/Ftq9-QDISmeXfYKwoZ3m4b2icBXh)
```python
class DoubleDQN:
    def learn(self):
        # 这一段和 DQN 一样:
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 这一段和 DQN 不一样
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:   # 如果是 Double DQN
            max_act4next = np.argmax(q_eval4next, axis=1)        # q_eval 得出的最高奖励动作
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN 选择 q_next 依据 q_eval 选出的动作
        else:       # 如果是 Natural DQN
            selected_q_next = np.max(q_next, axis=1)    # natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next


        # 这下面和 DQN 一样:
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
```
## 2.2 记录Q值
为了记录下我们选择动作时的Q值，接下来我们就修改``choose_action()``功能，让他记录下每次选择的Q值。
```python
class DoubleDQN:
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # 记录选的 Qmax 值
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # 随机选动作
            action = np.random.randint(0, self.n_actions)
        return action
```
## 2.3对比结果
接下来我们就来对比Natural DQN和Double DQN带来的不同结果，注意现在小棒子的动作是连续的，我们要把他离散化方便观看。
```python
import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env.seed(1) # 可重复实验
MEMORY_SIZE = 3000
ACTION_SPACE = 11    # 将原本的连续动作分离成 11 个动作

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # 在 [-2 ~ 2] 内离散化动作

        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize 到这个区间 (-1, 0). 立起来的时候 reward = 0.
        # 立起来以后的 Q target 会变成 0, 因为 Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # 所以这个状态时的 Q 值大于 0 时, 就出现了 overestimate.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
    return RL.q # 返回所有动作 Q 值

# train 两个不同的 DQN
q_natural = train(natural_DQN)
q_double = train(double_DQN)

# 出对比图
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
```
对比图：
![](http://puiarp73w.bkt.clouddn.com/Fnvaes8NOhMVUjegf65K2D7-9jxh)
可以看出，Natural DQN学的差不多的时候，在立起来时，大部分时间都是 估计的 Q值 要大于0, 这时就出现了 overestimate, 而 Double DQN 的 Q值 就消除了一些 overestimate, 将估计值保持在 0 左右.（小部分还有超过0是因为初始化参数的时候是随机的）

完整代码：[https://github.com/cristianoc20/RL_learning/tree/master/Double_DQN](https://github.com/cristianoc20/RL_learning/tree/master/Double_DQN)
参考：[https://github.com/MorvanZhou](https://github.com/MorvanZhou)
