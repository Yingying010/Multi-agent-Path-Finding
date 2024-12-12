import random
import time

from brain import DQN
from Record import ReplayBuffer
from Environmnet import Env
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# fixed random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用 GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Agent(DQN):
    def __init__(self, input_size, hidden_size, output_size, epsilon_delay, min_epsilon, gamma,
                 device, learning_rate, update_step=5):
        super().__init__(input_size, hidden_size, output_size, epsilon_delay, min_epsilon, gamma,
                         device, learning_rate, update_step)

        # save its attributions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epsilon_delay = epsilon_delay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.update_step = update_step
        

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        action_values = self.get_q_value(state)

        if random.uniform(0, 1) < self.epsilon or torch.all(action_values == action_values[0][0]):
            action = random.choice(range(4))
        else:
            action = action_values.max(1)[1].item()

        # 返回智能体选择的动作（0 到 3 之间的整数，代表不同动作）
        return action

    def test(self):
        # 设置 epsilon 值为 0.1，降低探索的比例，更多地利用训练好的策略
        self.epsilon = 0.1
        self.q_net.eval()


def train():
    agent = Agent(input_size, hidden_size, output_size, epsilon_decay_factor, min_epsilon, gamma, device, learning_rate, update_steps)
    env = Env()
    replay_buffer = ReplayBuffer(capacity)
    reward_positive_count = 0
    success_count = 0
    
    rewards = []  # 每回合的总奖励
    steps = []    # 每回合的总步数
    all_episode_paths = []  # 保存所有 episode 的路径数据
    


    for episode in range(episodes):
        state = env.reset()
        is_terminate = True
        total_reward = 0  # 用于累计当前回合的总奖励
        step_count = 0    # 记录当前回合的步数
        success_rate = success_count / (episode + 1)
        current_episode_path = []


        while is_terminate:

            action = agent.choose_action(state)
            next_state, action, reward, done, info = env.step(action)

            # Accumulate rewards and steps
            total_reward += reward
            step_count += 1
            current_episode_path.append(state)  # record the path

            if info["end"]:
                replay_buffer.add_important(state, action, reward, next_state, done)
                is_terminate = False
                print(f"-----------------Episode {episode}!!! Spent {env.step_counter} steps, reward is {env.reward}-------------------")
                if env.reward > 0:
                    reward_positive_count += 1
                else:
                    reward_positive_count -= 1
                    reward_positive_count = max(reward_positive_count, 0)
            elif info["target"]:
                replay_buffer.add_important(state, action, reward, next_state, done)
                is_terminate = False
                print(f"-----------------Episode {episode}!!! Spent {env.step_counter} steps, reward is {env.reward}-------------------")
                success_count +=1
                
            else:
                replay_buffer.add_common(state, action, reward, next_state, done)

            env.render()
            state = next_state

            if len(replay_buffer) > min_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # constructing training set
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'done': d
                }
                agent.update(transition_dict)

        # Record the rewards and steps for each episode
        rewards.append(total_reward)
        steps.append(step_count)

        # Early stopping conditions
        if success_rate > 0.8 and len(rewards) > 100 and sum(rewards[-100:]) / 100 > -50:
            break
        
        #Save the path of the current episode
        all_episode_paths.append(current_episode_path)  

    env.close()
    
    
    return agent, rewards, steps, all_episode_paths


def visualize_all_paths(grid, all_episode_paths, obstacles, start, goal):
    """
    可视化多个 episode 的综合路径
    :param grid: 网格环境 (二维数组)
    :param all_episode_paths: 多个 episode 的路径列表
    :param obstacles: 障碍物位置列表
    :param start: 起点 (x, y)
    :param goal: 终点 (x, y)
    """
    plt.figure(figsize=(8, 8))

    # 绘制网格
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if (x, y) in obstacles:
                plt.fill_between([y, y + 1], x, x + 1, color="black")  # 障碍物
            elif (x, y) == start:
                plt.fill_between([y, y + 1], x, x + 1, color="green")  # 起点
            elif (x, y) == goal:
                plt.fill_between([y, y + 1], x, x + 1, color="red")  # 终点
            else:
                plt.fill_between([y, y + 1], x, x + 1, color="white")  # 普通格子

    # 绘制所有路径
    for episode_idx, episode_path in enumerate(all_episode_paths):
        if episode_path:
            x_coords, y_coords = zip(*episode_path)
            plt.plot(y_coords, x_coords, alpha=0.3, linewidth=1)

    plt.title("Combined Paths Across Episodes")
    plt.axis("equal")
    plt.gca().invert_yaxis()  # 将坐标轴调整为左上角为原点
    plt.legend()
    plt.show()


def test(agent):
    env = Env()
    agent.test()
    state = env.reset()

    step_count = 0
    total_reward = 0
    while True:
        action = agent.choose_action(state)
        next_state, action, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward
        state = next_state
        env.render()
        time.sleep(0.2)

        if done:
            print(f"Test completed in {step_count} steps, Total reward: {total_reward}")
            if info.get('reached_goal', False):
                print("Successfully reached the goal!")
            else:
                print("Did not reach the goal.")
            break
        
    
def plot_learning_trend(rewards, steps):
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label="Total Reward per Episode", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Reward Trend Over Episodes")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episodes, steps, label="Steps per Episode", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Number of Steps")
    plt.title("Learning Steps Trend Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.show()
 

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    capacity = 5000
    input_size = 2
    hidden_size = 128
    output_size = 4
    min_epsilon = 0.1 
    epsilon_decay_factor = 0.995  # Decay factor

    gamma = 0.9
    learning_rate = 0.001
    episodes = 6000

    update_steps = 100
    min_size = 256
    batch_size = 128


    agent, rewards, steps, all_episode_paths  = train()
    
    # torch.save({
    #     'q_net_state_dict': agent.q_net.state_dict(),  # 保存 Q 网络权重
    #     'target_q_net_state_dict': agent.target_q_net.state_dict(),  # 保存目标网络权重
    #     'optimizer_state_dict': agent.optimizer.state_dict(),  # 保存优化器状态
    #     'epsilon': agent.epsilon,
    #     'gamma': agent.gamma,
    #     'learning_rate': agent.learning_rate,  # 保存学习率
    #     'update_step': agent.update_step,
    # }, 'model.pth')
    torch.save(agent, 'model.pth')


    plot_learning_trend(rewards, steps)
    
    env = Env()
    
    visualize_all_paths(
        grid=env.grid,
        all_episode_paths=all_episode_paths,
        obstacles=env.obstacles,
        start=env.start,
        goal=env.goal
    )

    print("Training complete. Next, test whether the agent can reach the target point.")
    

    # 测试
    
    test(agent)

    print("Test Completely")














# from brain import QTableAgent
# from Environmnet import Env
# import matplotlib.pyplot as plt
# import numpy as np


# def train():
#     env = Env()
#     agent1 = QTableAgent(env.maze.shape)
#     agent2 = QTableAgent(env.maze.shape)

#     episodes = 1000
#     for episode in range(episodes):
#         state1, state2 = env.reset_two_agents()
#         done1, done2 = False, False

#         while not (done1 and done2):
#             # Agent 1 行动
#             action1 = agent1.choose_action(state1)
#             next_state1, reward1, done1 = env.step_agent(0, action1)
#             agent1.update(state1, action1, reward1, next_state1, done1)
#             state1 = next_state1

#             # Agent 2 行动
#             action2 = agent2.choose_action(state2)
#             next_state2, reward2, done2 = env.step_agent(1, action2)
#             agent2.update(state2, action2, reward2, next_state2, done2)
#             state2 = next_state2

#         print(f"Episode {episode} complete: Agent1 Reward = {reward1}, Agent2 Reward = {reward2}")

#     # 保存训练好的模型
#     agent1.save_model("agent1_qtable.npy")
#     agent2.save_model("agent2_qtable.npy")

#     print("Training complete and models saved!")
#     return agent1, agent2

# def test_trained_agents():
#     # 加载环境和智能体
#     env = Env()
#     agent1 = QTableAgent(env.maze.shape)
#     agent2 = QTableAgent(env.maze.shape)

#     # 加载模型
#     agent1.load_model("agent1_qtable.npy")
#     agent2.load_model("agent2_qtable.npy")

#     # 测试逻辑
#     state1, state2 = env.reset_two_agents()
#     done1, done2 = False, False

#     while not (done1 and done2):
#         action1 = agent1.choose_action(state1)
#         next_state1, _, done1 = env.step_agent(0, action1)
#         state1 = next_state1

#         action2 = agent2.choose_action(state2)
#         next_state2, _, done2 = env.step_agent(1, action2)
#         state2 = next_state2

#         # 更新图像
#         env.update()

#     print("Testing completed!")
#     env.mainloop()  # 启动 Tkinter 主事件循环

# if __name__ == '__main__':
#     # 训练并保存模型
#     print("Starting training...")
#     train()

#     # 测试并显示图像
#     print("Testing trained agents...")
#     test_trained_agents()
