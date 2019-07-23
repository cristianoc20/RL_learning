from maze_env import Maze
from DQN_modified import DeepQNetwork


def run_maze():
    step = 0#用来控制什么时候学习
    for episode in range(25000):
        # 初始化环境
        observation = env.reset()

        while True:
            # 刷新环境
            env.render()

            # DQN根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个state,reward,是否终止
            observation_, reward, done = env.step(action)

            #DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)

            #控制学习起始时间和频率（选择200步之后再每5步学习一次的原因是先累积一些记忆再开始学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,#每200步替换一次target_net的参数
                      memory_size=2000, #记忆上线
                      # output_graph=True #是否输出tensorboard文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()