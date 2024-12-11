import pybullet as p
import time
import pybullet_data
import yaml
from cbs import cbs
import math
import threading

import random

class RRT:
    def __init__(self, start, goal, obstacles, rand_area, expand_dis=1.0, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.rand_area = rand_area
        self.expand_dis = expand_dis
        self.max_iter = max_iter
        self.node_list = [self.start]

    def planning(self):
        for _ in range(self.max_iter):
            rnd = self.get_random_point()
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            new_node = self.steer(nearest_node, rnd)

            if self.check_collision(new_node, nearest_node):
                self.node_list.append(new_node)

                if self.is_near_goal(new_node):
                    final_node = self.steer(new_node, [self.goal.x, self.goal.y])
                    if self.check_collision(final_node, new_node):
                        return self.get_path(final_node)
        return None

    def get_random_point(self):
        return [random.uniform(self.rand_area[0], self.rand_area[1]), random.uniform(self.rand_area[0], self.rand_area[1])]

    def get_nearest_node(self, node_list, rnd):
        return min(node_list, key=lambda node: (node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2)

    def steer(self, from_node, to_point):
        angle = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        new_node = Node(from_node.x + self.expand_dis * math.cos(angle),
                        from_node.y + self.expand_dis * math.sin(angle))
        new_node.parent = from_node
        return new_node

    def check_collision(self, node, from_node):
        for obs in self.obstacles:
            if self.line_intersects_rect([from_node.x, from_node.y], [node.x, node.y], obs):
                return False
        return True

    def is_near_goal(self, node):
        return math.hypot(node.x - self.goal.x, node.y - self.goal.y) <= self.expand_dis

    def get_path(self, goal_node):
        path = [[self.goal.x, self.goal.y]]
        while goal_node.parent is not None:
            path.append([goal_node.x, goal_node.y])
            goal_node = goal_node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]

    def line_intersects_rect(self, start, end, rect):
        x1, y1 = start
        x2, y2 = end
        ox_min, oy_min, ox_max, oy_max = rect
        return (
            self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_min, ox_max, oy_min) or
            self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_max, ox_max, oy_max) or
            self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_min, ox_min, oy_max) or
            self.line_intersects_line(x1, y1, x2, y2, ox_max, oy_min, ox_max, oy_max)
        )

    def line_intersects_line(self, x1, y1, x2, y2, x3, y3, x4, y4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and
                ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4)))


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None



def create_boundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("./final_challenge/assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, -1, 0.5])


def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
            
    # Create env boundaries
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])

    # Create env obstacles
    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("./final_challenge/assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])
    return env_params


def create_agents(yaml_file):
    """
    Creates and loads turtlebot agents.

    Returns list of agent IDs and dictionary of agent IDs mapped to each agent's goal.
    """
    agent_box_ids = []
    box_id_to_goal = {}
    agent_name_to_box_id = {}
    with open(yaml_file, 'r') as f:
        try:
            agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
        
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
        agent_box_ids.append(box_id)
        box_id_to_goal[box_id] = agent["goal"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params


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

def spin(agent):
    """
    Make the agent spin in place for a full circle.

    Args:
        agent (int): Agent ID.
    """
    spin_velocity = 2.0  # radians per second
    spin_time = 2 * math.pi / spin_velocity  # Time to complete a full circle

    start_time = time.time()
    while time.time() - start_time < spin_time:
        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=-spin_velocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=spin_velocity, force=1)
        time.sleep(0.01)

    # 停止机器人
    p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
    p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
    print(f"Agent {agent} completed spinning.")


def navigation(agent, goal, schedule):
    """
        Set velocity for robots to follow the path in the schedule.

        Args:

        agents: array containing the IDs for each agent

        schedule: dictionary with agent IDs as keys and the list of waypoints to the goal as values

        index: index of the current position in the path.

        Returns:

        Leftwheel and rightwheel velocity.
    """
    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.4
    while(not checkPosWithBias(basePos[0], goal, dis_th)):
        basePos = p.getBasePositionAndOrientation(agent)
        next = [schedule[index]["x"], schedule[index]["y"]]
        if(checkPosWithBias(basePos[0], next, dis_th)):
            # # 检查是否到达触发点
            # if index == 10:
            #     print(f"Agent {agent} spinning at waypoint {index}.")
            #     spin(agent)  # 调用转圈函数
            index = index + 1
        if(index == len(schedule)):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))
        
        if index + 1 < len(schedule):
            next_target = schedule[index + 1]
            next_direction = math.atan2(next_target["y"] - schedule[index]["y"], next_target["x"] - schedule[index]["x"])
            turn_angle = abs(next_direction - goal_direction)

            if turn_angle > math.pi:
                turn_angle = 2 * math.pi - turn_angle
        else:
            turn_angle = 0  
            
            

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
        k1, k2 = 50, 20
        linear = k1 * math.cos(theta)
        angular = k2 * theta
        
        max_linear_speed = 50.0 
        max_angular_speed = 7.0 
        
        print(f"angular:{angular} linear{linear}")
        if abs(angular)>10:
            linear = 5
            
        # 在拐弯点减速
        # print(f"turn_angle:{turn_angle}")
        if turn_angle > 1.5:
            linear *= 0.65  # 减速

        linear = max(-max_linear_speed, min(max_linear_speed, linear))
        angular = max(-max_angular_speed, min(max_angular_speed, angular))


        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular
        
        

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        # time.sleep(0.001)
    print(agent, "here")


def run(agents, goals, schedule):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    for agent in agents:
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


# physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=30')
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

plane_id = p.loadURDF("plane.urdf")

global env_loaded
env_loaded = False

# Create environment
env_params = create_env("./final_challenge/env.yaml")

# Create turtlebots
agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("./final_challenge/singleActor_noFetchPoint.yaml")

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[4.5, 4.5, 4])


paths = {}
for agent in agent_yaml_params["agents"]:
    start = agent["start"]
    goal = agent["goal"]
    rrt = RRT(start, goal, env_params["map"]["obstacles"], rand_area=[0, env_params["map"]["dimensions"][0]])
    path = rrt.planning()
    if path:
        paths[agent["name"]] = [{"x": p[0], "y": p[1]} for p in path]
        output = dict()
        output["schedule"] = path
        with open("./final_challenge/RRT_output.yaml", 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)
    else:
        print(f"Agent {agent['name']} could not find a path!")
# cbs_schedule = read_cbs_output("./final_challenge/cbs_output_noFetchPoint_singleAgent.yaml")
# Replace agent name with box id in cbs_schedule
# box_id_to_schedule = {}
# for name, value in cbs_schedule.items():
#     box_id_to_schedule[agent_name_to_box_id[name]] = value

# run(agent_box_ids, box_id_to_goal, box_id_to_schedule)
time.sleep(2)