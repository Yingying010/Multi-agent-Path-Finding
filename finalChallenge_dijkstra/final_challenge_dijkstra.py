import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import yaml
import heapq
import math

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from finalChallenge_dijkstra.dijkstra_cbs import run as dijkstra_run
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
        p.loadURDF("assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("assets/cube.urdf", [-1, -1, 0.5])

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


def run(agents, goals, schedule):
    threads = []
    for agent in agents.values():
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()



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

env_params = create_env("environment/env.yaml")

agent_ids, agent_starts, agent_goals, agent_yaml_params = create_agents("environment/multiActors_noFetchPoint.yaml")

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                    cameraTargetPosition=[4.5, 4.5, 4])

map_params, extra_obstacles = env_params  # 解包元组

dimensions = map_params["map"]["dimensions"]
obstacles = map_params["map"]["obstacles"] + extra_obstacles  # 合并障碍物
# 获取output.yaml
dijkstra_run(
    dimensions=dimensions,
    obstacles=obstacles,
    agents=agent_yaml_params["agents"],
    out_file="finalChallenge_dijkstra/output/dijkstra_output.yaml",
)
cbs_schedule = read_cbs_output("finalChallenge_dijkstra/output//dijkstra_output.yaml")

print(f"schedule:{cbs_schedule}")

agent_schedules = {}
for name, value in cbs_schedule.items():
    print(name)
    print(agent_ids[name])
    agent_schedules[agent_ids[name]] = value

run(agent_ids, agent_goals, agent_schedules)
time.sleep(2)

