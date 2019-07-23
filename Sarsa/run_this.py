"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # 初始化环境
        observation = env.reset()

        # Sarsa根据state观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为，获得下一个state_(observation_),reward，和终止信号
            observation_, reward, done = env.step(action)

            # 根据下一个state(observation_)选取下一个action_
            action_ = RL.choose_action(str(observation_))

            #从(s, a, r, s, a)中学习，更新Q_table的参数
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个的observation_和action_当成对应下一步的参数
            observation = observation_
            action = action_


            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
