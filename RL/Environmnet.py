import tkinter as tk
import random
import numpy as np
import copy
import time
import heapq

maze_array = np.array(
    [   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1],
        [1, 0, 0, 1, 1, 1, 1, 1, 0, 0 ,1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
)

print(f"Maze array:{maze_array}")
whitegrids = np.transpose(np.where(maze_array == 0)).tolist()
barriers = np.transpose(np.where(maze_array == 1)).tolist()
target = np.transpose(np.where(maze_array == 2)).tolist()
start_pos = tuple(np.transpose(np.where(maze_array == 3))[0])
dimensions = (11,11)
target_pos = tuple(target[0])

print(barriers, start_pos, target, sep="\n")

row, column = maze_array.shape
row_sep = 50
column_sep = 50

color_map = {0: 'white', 1: 'black', 2: "yellow", 3: "red"}

def a_star(grid, start, goal):
    """A*算法实现"""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    # 从起点到当前节点的实际代价（步数）
    g_score = {start: 0}
    # 作为优先级的总估计代价
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return None


def create_grid_map(dimensions, obstacles, resolution=1.0):
    """创建栅格地图"""
    width, height = int(dimensions[0] / resolution), int(dimensions[1] / resolution)
    grid = np.zeros((width, height))
    for obs in obstacles:
        x, y = int(obs[0] / resolution), int(obs[1] / resolution)
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] = 1  # 将障碍物标记为1
    return grid


def searchSchedule(start, goal, obstacles, dimensions):
    schedule = []
    grid = create_grid_map(dimensions, obstacles, resolution=1.0)
    print(grid)
    start_grid = (int(start[0]), int(start[1]))
    goal_grid = (int(goal[0]), int(goal[1]))

    if grid[start_grid] == 1 or grid[goal_grid] == 1:
        print("Start or goal is inside an obstacle!")
        return

    schedule = a_star(grid, start_grid, goal_grid)

    if schedule is None:
        print("No path found!")
        return

    print(f"Path found: {schedule}")
    return schedule


class Env(tk.Tk):
    def __init__(self):
        super().__init__()
        self.maze = copy.copy(maze_array)
        self.widget_dict = {}
        self.step_counter = 0
        self.env_info = {"end": False, "action": None, "target": False}
        self.state = copy.copy(start_pos)
        self.reward = 0
        self.shortest_path = searchSchedule(start_pos, target_pos, barriers, dimensions)
        self.visited = {}  # 记录访问状态的次数
        self.history_path = []  # 存储历史路径
        self.forbidden_states = set()  # 记录禁区状态
        self.current_path_index = 0  # 初始化路径索引
        self.visited_edges = {}  # 使用字典记录路径段及其访问次数
        self.last_state = copy.copy(self.state) # 初始状态
        self.obstacles = barriers  # 确保障碍物被存储为环境的一部分
        self.start = start_pos
        self.goal = target_pos
        self.grid = [[0 for _ in range(10)] for _ in range(10)]  # 示例
        self.visited_states = set() 
        self.distance = 0

        self.focus_pos()
        self.create_widget()

    def focus_pos(self):
        # 设置窗口大小
        window_width = (column + 1) * column_sep
        window_height = (row + 1) * row_sep
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_widget(self):
        state_show_label = tk.Label(self, text="", fg="black")
        state_show_label.pack()
        self.widget_dict[2] = state_show_label

        width = column * column_sep
        height = row * row_sep
        board = tk.Canvas(self, width=width, height=height, bg='white')
        board.pack()
        self.widget_dict[1] = board
        self.generate_maze()

    def reset(self):
        self.update()
        self.state = copy.copy(start_pos)
        self.maze = copy.deepcopy(maze_array)
        self.env_info = {"end": False, "action": None, "target": None}
        self.step_counter = 0
        self.reward = 0
        self.history_path = []  # 清空历史路径
        self.last_state = copy.copy(start_pos)  # 回到起始点
        self.visited_edges = {}  # 清空路径段访问记录
        self.distance = np.sqrt((self.state[0]-target_pos[0])**2 + (self.state[1]-target_pos[1])**2)
        self.render()

        return copy.copy(self.state)
    
    def is_subpath(self, history_path, shortest_path):
        """
        检查历史路径是否完全包含在最短路径中
        :param history_path: 当前走过的路径列表
        :param shortest_path: A*计算出的最短路径
        :return: True / False
        """
        history_set = set(history_path)
        for i in range(len(shortest_path) - len(history_path) + 1):
            if set(shortest_path[i:i+len(history_path)]) == history_set:
                return True
        return False

    def step(self, action):
        "action: 0:left 1:right 2:up 3:down"
        # (x0-t0)^2 - (x1-t1)^2

        prev_state = copy.copy(self.state)


        '''
        start the strategy
        '''
        # initial reward
        reward = 0
        
        
        self.step_counter += 1
        done = False
        self.maze[tuple(self.state)] = 0
        self.env_info["action"] = action
                
        # 执行动作更新位置
        self.state = list(self.state)
        if action == 0 and self.state[1] > 0:  # 向左移动
            self.state[1] -= 1
            print("move left")
        elif action == 1 and self.state[1] < column - 1:  # 向右移动
            self.state[1] += 1
            print("move right")
        elif action == 2 and self.state[0] > 0:  # 向上移动
            self.state[0] -= 1
            print("move up")
        elif action == 3 and self.state[0] < row - 1:  # 向下移动
            self.state[0] += 1
            print("move down")
        else:
            raise Exception("action range in 0-3")

        
        # add it in history_path
        self.history_path.append(tuple(self.state))
        
        # goal position -- end 
        if self.state in target:
            self.env_info["end"] = True
            self.env_info["reached_goal"] = True
            done = True
            
            # 固定目标奖励
            base_goal_reward = 500
            
            # 剩余步数奖励（适当降低放大因子）
            remaining_steps_bonus =  100* (50 - self.step_counter)
            
            # 路径奖励的比例调整
            reward += base_goal_reward + remaining_steps_bonus
            
            print(f"arrive the goal position: {reward}")

        # barriers -- end
        if self.state in barriers:
            self.env_info["end"] = True
            done = True
            reward -= 500
            print(f"Encountered an obstacle  -{500}")

        # 基于距离变化给予奖励或惩罚
        # 计算距离
        current_distance = np.sqrt((self.state[0] - target_pos[0])**2 + (self.state[1] - target_pos[1])**2)
        if len(self.history_path) > 1 and self.state not in barriers : 
            print(f"prev_state: {prev_state}  current state:{self.state}")
            prev_distance = np.sqrt((self.history_path[-2][0] - target_pos[0])**2 + (self.history_path[-2][1] - target_pos[1])**2)
            
            if current_distance < prev_distance:
                reward += 100  # 靠近目标点的奖励
                print(f"Moving closer to the target: reward +{100}")
            else:
                reward -= 20  # 远离目标点的惩罚
                print(f"Moving away from the target: penalty -{20}")

  
        print(f"current state:{tuple(self.state)}")
        print(f"self.history_path:{self.history_path}")
        if len(self.history_path)>=2 and tuple(self.state) in self.history_path[:-2]:
            self.env_info["end"] = True
            done = True
            reward -= 500
            print(f"The current position already exists in the historical path, penalty -{500}, terminated!")

        # role movement
        self.maze[tuple(self.state)] = 3
        # remove calculation
        self.reward += reward

        "返回observation action reward done env_info"
        return copy.copy(self.state), action, reward, done, copy.deepcopy(self.env_info)

    def render(self):
        # 用于渲染环境
        self.generate_maze()
        if self.env_info["end"]:
            font_color = 'red'
        else:
            font_color = 'black'
        self.widget_dict[2].config(text=f"steps: {self.step_counter} action: {self.env_info['action'] } rewards: {self.reward}", fg=font_color)
        self.update()

        if self.env_info["end"]:
            time.sleep(0.2)
            self.reset()

    def close(self):
        self.destroy()

    def generate_maze(self):
        board = self.widget_dict[1]
        board.delete("all")
        for column_idx, maze_row in enumerate(self.maze):
            for row_idx, item in enumerate(maze_row):
                start_pos = (row_idx * row_sep, column_idx * column_sep)
                end_pos = ((row_idx + 1) * row_sep, (column_idx + 1) * column_sep)
                if item == 2:
                    board.create_rectangle(start_pos, end_pos, fill='white')
                    board.create_oval(start_pos, end_pos, fill=color_map[item])
                    continue
                else:
                    board.create_rectangle(start_pos, end_pos, fill=color_map[item])

    def random_walk(self, steps=10):
        for i in range(steps):
            action = random.choice(range(4))
            self.step(action)
            self.render()
            time.sleep(3000)

# import tkinter as tk
# import copy
# import numpy as np

# # 定义迷宫布局
# maze_array = np.array(
#     [
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
#         [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#         [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#         [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#         [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 2, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     ]
# )

# # 颜色映射
# color_map = {
#     0: "white",  # 可行走路径
#     1: "black",  # 障碍
#     2: "red",    # 目标
# }

# # 每个格子的宽度和高度
# row_sep = 50
# column_sep = 50

# class Env(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.maze = copy.copy(maze_array)
#         self.widget_dict = {}

#         # 智能体的起点和目标点
#         self.start_positions = [(1, 1), (8, 8)]
#         self.target_positions = [(9, 8), (2, 1)]

#         # 记录智能体的当前位置
#         self.agent_positions = list(self.start_positions)
#         self.reward = [0, 0]  # 每个智能体的奖励
#         self.done = [False, False]  # 每个智能体是否完成

#         self.center_window()
#         self.create_widget()

#     def center_window(self):
#         """设置窗口居中"""
#         window_width = self.maze.shape[1] * column_sep
#         window_height = self.maze.shape[0] * row_sep
#         screen_width = self.winfo_screenwidth()
#         screen_height = self.winfo_screenheight()
#         x = int((screen_width - window_width) / 2)
#         y = int((screen_height - window_height) / 2)
#         self.geometry(f"{window_width}x{window_height}+{x}+{y}")

#     def create_widget(self):
#         """创建画布"""
#         state_show_label = tk.Label(self, text="", fg="black")
#         state_show_label.pack()
#         self.widget_dict[2] = state_show_label

#         width = self.maze.shape[1] * column_sep
#         height = self.maze.shape[0] * row_sep
#         board = tk.Canvas(self, width=width, height=height, bg="white")
#         board.pack()
#         self.widget_dict[1] = board
#         self.render()

#     def reset_two_agents(self):
#         """重置环境"""
#         self.agent_positions = list(self.start_positions)  # 重置智能体位置
#         self.reward = [0, 0]
#         self.done = [False, False]
#         self.render()
#         return tuple(self.agent_positions)

#     def step_agent(self, agent_id, action):
#         """执行智能体的动作"""
#         if self.done[agent_id]:
#             return self.agent_positions[agent_id], 0, True

#         x, y = self.agent_positions[agent_id]

#         # 根据动作更新位置
#         if action == 0 and self.maze[x, y - 1] != 1:  # 左
#             y -= 1
#         elif action == 1 and self.maze[x, y + 1] != 1:  # 右
#             y += 1
#         elif action == 2 and self.maze[x - 1, y] != 1:  # 上
#             x -= 1
#         elif action == 3 and self.maze[x + 1, y] != 1:  # 下
#             x += 1

#         new_position = (x, y)

#         # 检查是否与另一个智能体重叠
#         if new_position in self.agent_positions:
#             reward = -10  # 碰撞惩罚
#             print(f"Collision detected for agent {agent_id + 1}")
#         elif new_position == self.target_positions[agent_id]:
#             reward = 100  # 到达目标点
#             self.done[agent_id] = True
#         else:
#             reward = -1  # 移动惩罚

#         # 更新智能体位置
#         self.agent_positions[agent_id] = new_position
#         self.render()
#         return new_position, reward, self.done[agent_id]

#     def render(self):
#         """渲染环境"""
#         board = self.widget_dict[1]
#         board.delete("all")
#         for column_idx, maze_row in enumerate(self.maze):
#             for row_idx, item in enumerate(maze_row):
#                 start_pos = (row_idx * column_sep, column_idx * row_sep)
#                 end_pos = ((row_idx + 1) * column_sep, (column_idx + 1) * row_sep)

#                 # 绘制基础网格
#                 if item == 2:  # 目标点
#                     board.create_rectangle(start_pos, end_pos, fill="white")
#                     board.create_oval(start_pos, end_pos, fill=color_map[item])
#                 else:
#                     board.create_rectangle(start_pos, end_pos, fill=color_map[item])

#         # 绘制智能体
#         for agent_idx, position in enumerate(self.agent_positions):
#             x, y = position
#             start_pos = (y * column_sep, x * row_sep)
#             end_pos = ((y + 1) * column_sep, (x + 1) * row_sep)
#             color = "yellow" if agent_idx == 0 else "blue"  # 区分智能体颜色
#             board.create_oval(start_pos, end_pos, fill=color)


# if __name__ == '__main__':
#     env = Env()  # 创建环境实例

#     def step_randomly():
#         """随机执行智能体动作以测试环境"""
#         for _ in range(100):  # 随机运行 20 步
#             action1 = np.random.choice(4)  # 智能体 1 的随机动作
#             action2 = np.random.choice(4)  # 智能体 2 的随机动作

#             state1, reward1, done1 = env.step_agent(0, action1)
#             state2, reward2, done2 = env.step_agent(1, action2)

#             print(f"Agent 1: State={state1}, Reward={reward1}, Done={done1}")
#             print(f"Agent 2: State={state2}, Reward={reward2}, Done={done2}")
#             env.update()  # 更新 Tkinter 界面

#     # 启动随机测试
#     env.after(1000, step_randomly)  # 延迟 1 秒后开始随机动作
#     env.mainloop()  # 启动 Tkinter 主事件循环