"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # 初始化state的observation
        observation = env.reset()

        while True:
            # 更新环境
            env.render()

            # RL大脑根据state的observation选择action
            action = RL.choose_action(str(observation))

            # RL实施action,并得到环境返回的下一个state的observation,reward和done(是否碰到黑箱或者宝藏)
            observation_, reward, done = env.step(action)

            # RL从这个序列中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将state的observation传递到下一个循环
            observation = observation_

            # 如果碰到黑箱或者宝藏则结束回合
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    #定义环境和RL方法
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    #可视化env
    env.after(100, update)
    env.mainloop()
