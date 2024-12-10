import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import yaml
import heapq
import math
from dijkstra_planner import run as dijkstra_run
import threading

# 生成栅格地图
def create_grid_map(dimensions, obstacles, resolution=1.0):
    """创建栅格地图"""
    width, height = int(dimensions[0] / resolution), int(dimensions[1] / resolution)
    grid = np.zeros((width, height))
    for obs in obstacles:
        x, y = int(obs[0] / resolution), int(obs[1] / resolution)
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] = 1  # 将障碍物标记为1
    return grid

# Dijkstra算法
def dijkstra(grid, start, goal):
    """使用 Dijkstra 算法在网格地图中寻找最短路径"""
    rows, cols = grid.shape
    visited = set()
    queue = []
    heapq.heappush(queue, (0, start))  # 优先队列，(cost, (x, y))
    came_from = {}  # 记录路径

    while queue:
        cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四方向邻居
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                new_cost = cost + 1
                if neighbor not in visited:
                    heapq.heappush(queue, (new_cost, neighbor))
                    came_from[neighbor] = current

    return None  # 未找到路径

def checkPosWithBias(Pos, goal, bias):
    """
        Check if pos is at goal with bias

        Args:

        Pos: Position to be checked, [x, y]

        goal: goal position, [x, y]

        bias: bias allowed

        Returns:

        True if pos is at goal, False otherwise
    """
    if(Pos[0] < goal[0] + bias and Pos[0] > goal[0] - bias and Pos[1] < goal[1] + bias and Pos[1] > goal[1] - bias):
        return True
    else:
        return False

def searchSchedule(start, goal, obstacles, dimensions):
    schedule = []
    grid = create_grid_map(dimensions, obstacles, resolution=1.0)
    start_grid = (int(start[0]), int(start[1]))
    goal_grid = (int(goal[0]), int(goal[1]))

    if grid[start_grid] == 1 or grid[goal_grid] == 1:
        print("Start or goal is inside an obstacle!")
        return

    schedule = dijkstra(grid, start_grid, goal_grid)

    if schedule is None:
        print("No path found!")
        return

    print(f"Path found: {schedule}")
    return schedule



# 机器人导航
def navigation(agent, goal, schedule):
    basePos = p.getBasePositionAndOrientation(agent)
    pos_index = 0
    dis_th = 0.4
    while(not checkPosWithBias(basePos[0], goal, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule[pos_index]["x"],schedule[pos_index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            pos_index = pos_index + 1
        if(pos_index == len(schedule)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[pos_index]["y"] - y), (schedule[pos_index]["x"] - x))

        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi
        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        current = [x, y]
        distance = math.dist(current, next)
        k1, k2, A = 20, 5, 20
        linear = k1 * math.cos(theta)
        angular = k2 * theta

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)


# 创建边界
def create_boundaries(length, width):
    for i in range(length):
        p.loadURDF("cube.urdf", [i, -1, 0.5])
        p.loadURDF("cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("cube.urdf", [-1, i, 0.5])
        p.loadURDF("cube.urdf", [length, i, 0.5])
    p.loadURDF("cube.urdf", [length, -1, 0.5])
    p.loadURDF("cube.urdf", [length, width, 0.5])
    p.loadURDF("cube.urdf", [-1, width, 0.5])
    p.loadURDF("cube.urdf", [-1, -1, 0.5])

# 创建环境
def create_env(yaml_file):
    obstacles = []
    with open(yaml_file, 'r') as f:
        env_params = yaml.load(f, Loader=yaml.FullLoader)
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])

    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("cube.urdf", [obstacle[0], obstacle[1], 0.5])
        obstacles.append([obstacle[0], obstacle[1]])
    return env_params, obstacles

# 创建机器人
def create_agents(yaml_file):
    ids = []
    agent_ids = {}
    agent_goals = {}
    agent_starts = {}
    with open(yaml_file, 'r') as f:
        agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)

    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
        ids.append(box_id)
        agent_goals[box_id] = agent["goal"]
        agent_starts[box_id] = agent["start"]
        agent_ids[agent["name"]] = box_id
    return agent_ids, agent_starts, agent_goals, agent_yaml_params


def read_cbs_output(file):
    """
        Read file from output.yaml, store path list.

        Args:

        output_yaml_file: output file from cbs.

        Returns:

        schedule: path to goal position for each robot.
    """
    with open(file, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return params["schedule"]

def run_multi_theads(agents, goals, schedule):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    for agent in agents.values():
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()




# 主函数
def main():
    # physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=Robot2_finalChanllege_dijkstra.mp4 --mp4fps=15')
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # Disable tinyrenderer, software (CPU) renderer, we don't use it here
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    plane_id = p.loadURDF("plane.urdf")

    global env_loaded
    env_loaded = False

    env_params, obstacles = create_env("final_challenge/env.yaml")
    dimensions = env_params["map"]["dimensions"]
    obstacles = np.array(env_params["map"]["obstacles"])

    agent_ids, agent_starts, agent_goals, agent_yaml_params = create_agents(r"F:\turtlebot_simulation1\final_challenge\multiActors_noFetchPoint.yaml")
    print(agent_ids)
    print(agent_starts)
    print(agent_goals)


    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -10)
    p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                        cameraTargetPosition=[4.5, 4.5, 4])

    # 获取output.yaml
    dijkstra_run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="final_challenge/dijkstra_output.yaml")
    schedule = read_cbs_output("final_challenge/dijkstra_output.yaml")

    print(f"schedule:{schedule}")

    agent_schedules = {}
    for name, value in schedule.items():
        print(name)
        print(agent_ids[name])
        agent_schedules[agent_ids[name]] = value

    run_multi_theads(agent_ids, agent_goals, agent_schedules)


if __name__ == "__main__":
    main()
