from Environmnet import Env
import time
import torch
from main import Agent

import numpy as np
np.random.seed(42)

def test(agent):
    env_test = Env()
    # print(env_test.grid)  
    agent.test()
    state = env_test.reset()
    isDone = False
    step_count = 0
    total_reward = 0
    while True:
        action = agent.choose_action(state)
        next_state, action, reward, done, info = env_test.step(action)
        step_count += 1
        total_loss = 0 #  # 用于累计测试损失
        total_reward += reward
        state = next_state
        env_test.render()
        time.sleep(0.2)

        if done:
            print(f"Test completed in {step_count} steps, Total reward: {total_reward}")
            if info.get('reached_goal', False):
                print("Successfully reached the goal!")
                isDone = True
            else:
                print("Did not reach the goal.")
                isDone = False

            total_loss += -total_reward
            break
    # 关闭环境窗口
    env_test.close()
        
    return isDone, total_loss

# # 模型文件路径
# model_path = 'model.pth'

# # 加载模型
# checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# # 检查模型类型和内容
# print(f"Type of loaded checkpoint: {type(checkpoint)}")
# if isinstance(checkpoint, dict):
#     print("Checkpoint is a state dictionary.")
#     print("Keys in the state dictionary:")
#     for key in checkpoint.keys():
#         print(f"- {key}")
# else:
#     print("Checkpoint is not a state dictionary.")
#     print(f"Loaded object type: {type(checkpoint)}")
    




# 加载状态字典
# model_path = 'model.pth'
# state_dict = torch.load(model_path, map_location=torch.device('cpu'))


# agent = Agent(
#     input_size=2,          # 根据你的任务设置输入大小
#     hidden_size=128,       # 根据你的网络设置隐藏层大小
#     output_size=4,         # 动作数量
#     epsilon_delay=1000,    # 如果需要设置
#     min_epsilon=state_dict['epsilon'],  # 加载保存的 epsilon
#     gamma=state_dict['gamma'],          # 加载保存的 gamma
#     device=torch.device('cpu'),         # 使用 CPU 或 GPU
#     learning_rate=state_dict['learning_rate'],  # 加载学习率
#     update_step=state_dict['update_step']       # 加载更新步数
# )

# print("Agent Attributes:")
# for attr in dir(agent):
#     if not attr.startswith("__") and not callable(getattr(agent, attr)):
#         print(f"{attr}: {getattr(agent, attr)}")

# agent.epsilon = 0.1  # 测试时设置低 epsilon

# # 加载 Q 网络和目标 Q 网络的参数
# agent.q_net.load_state_dict(state_dict['q_net_state_dict'])
# agent.target_q_net.load_state_dict(state_dict['target_q_net_state_dict'])

# print("Q Network Weights (Summary):")
# for name, param in agent.q_net.state_dict().items():
#     print(f"{name}: {param.mean().item():.4f}, {param.std().item():.4f}")

# print("Target Q Network Weights (Summary):")
# for name, param in agent.target_q_net.state_dict().items():
#     print(f"{name}: {param.mean().item():.4f}, {param.std().item():.4f}")

# # 如果需要加载优化器
# if hasattr(agent, 'optimizer'):
#     agent.optimizer.load_state_dict(state_dict['optimizer_state_dict'])





model = torch.load('model.pth')

# 切换到测试模式
model.test()

# 调用测试
NumberOfTest = 100
successCount = 0
failure_penalty = 100  # 如果失败，每次增加的损失
for i in range(NumberOfTest):
    print(f"===============Test{i}==================")
    result,total_loss = test(model)
    if result == True:
        successCount+=1
    
print(f"successCount:{successCount} NumberOfTest:{NumberOfTest}")

average_loss = total_loss / NumberOfTest
print(f"Average Test Loss: {average_loss:.4f}")
print(f"Success Rate: {successCount / NumberOfTest:.2%}")


