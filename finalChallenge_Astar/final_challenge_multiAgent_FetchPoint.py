import pybullet as p
import time
import pybullet_data
import yaml

import math
import threading

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import cbs.cbs as cbs


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

def create_env(yaml_file):
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])
    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])
    return env_params

def create_agents(yaml_file):
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
        # 将最终目标goal2存入，以供导航最终检测使用
        box_id_to_goal[box_id] = agent["goal2"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params

def read_cbs_output(file):
    with open(file, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return params["schedule"]

def checkPosWithBias(Pos, goal, bias):
    return (goal[0] - bias <= Pos[0] <= goal[0] + bias) and (goal[1] - bias <= Pos[1] <= goal[1] + bias)

def navigation(agent, goal, schedule, phase_1_length):
    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.4
    previous_left = 0
    previous_right = 0
    max_velocity_change = 1.0

    while not checkPosWithBias(basePos[0], goal, dis_th):
        basePos = p.getBasePositionAndOrientation(agent)
        if index >= len(schedule):
            # 已无更多waypoint
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break
        next_point = [schedule[index]["x"], schedule[index]["y"]]
        if checkPosWithBias(basePos[0], next_point, dis_th):
            index += 1
            # 当刚刚完成第一阶段路径时暂停2秒
            if index == phase_1_length:
                # 停止机器人
                p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                # 等待2秒
                time.sleep(2)
            if index == len(schedule):
                p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                break
            continue

        x, y = basePos[0][0], basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))

        # 规范化角度到[0,2*pi)
        if(Orientation < 0):
            Orientation = Orientation + 2 * math.pi
        if(goal_direction < 0):
            goal_direction = goal_direction + 2 * math.pi
        theta = goal_direction - Orientation

        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi

        k1, k2 = 30, 15
        linear = k1 * math.cos(theta)
        angular = k2 * theta
        
        
        max_angular_speed = 12.0 
        
        if angular>1:
            linear = 5
            
        if agent==72:
            max_linear_speed = 20.0
        else:
            max_linear_speed = 100.0

        linear = max(-max_linear_speed, min(max_linear_speed, linear))
        angular = max(-max_angular_speed, min(max_angular_speed, angular))


        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular
        
        
        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
        time.sleep(0.01)
    print(agent, "reached goal")

def run(agents, goals, schedule, phase_1_lengths):
    threads = []
    for agent in agents:
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent], phase_1_lengths[agent]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    import os

    # physics_client = p.connect(p.GUI)
    
    # 分辨率设置
    width = 1920
    height = 1080
    physicsClient = p.connect(p.GUI)
    # physicsClient = p.connect(p.GUI, options="--start_demo_name=Physics Server --width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=30")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    plane_id = p.loadURDF("plane.urdf")

    env_params = create_env("environment/env.yaml")
    agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("environment/multiActors_fetchPoint.yaml")

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -10)
    p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[4.5, 4.5, 4])

    # 阶段1：从start到goal1
    agents_phase_1 = []
    for agent in agent_yaml_params["agents"]:
        agents_phase_1.append({
            "name": agent["name"],
            "start": agent["start"],
            "goal": agent["goal1"]
        })

    print("Running CBS Phase 1 (start→goal1)...")
    cbs.run(dimensions=env_params["map"]["dimensions"],
            obstacles=env_params["map"]["obstacles"],
            agents=agents_phase_1,
            out_file="finalChallenge_Astar/output/cbs_output_phase_1_multi.yaml")
    cbs_schedule_phase_1 = read_cbs_output("finalChallenge_Astar/output/cbs_output_phase_1_multi.yaml")
    print("Phase 1 schedule:", cbs_schedule_phase_1)

    # 更新起点为goal1
    for agent in agent_yaml_params["agents"]:
        agent["start"] = agent["goal1"]

    # 阶段2：从goal1到goal2
    agents_phase_2 = []
    for agent in agent_yaml_params["agents"]:
        agents_phase_2.append({
            "name": agent["name"],
            "start": agent["start"],
            "goal": agent["goal2"]
        })

    print("Running CBS Phase 2 (goal1→goal2)...")
    cbs.run(dimensions=env_params["map"]["dimensions"],
            obstacles=env_params["map"]["obstacles"],
            agents=agents_phase_2,
            out_file="finalChallenge_Astar/output/cbs_output_phase_2_multi.yaml")

    cbs_schedule_phase_2 = read_cbs_output("finalChallenge_Astar/output/cbs_output_phase_2_multi.yaml")
    print("Phase 2 schedule:", cbs_schedule_phase_2)

    if not cbs_schedule_phase_2:
        print("No valid path found in Phase 2. Agents will stop at goal1.")
        exit()

    # 合并两阶段路径
    combined_schedule = {}
    phase_1_lengths = {}
    for agent in agent_yaml_params["agents"]:
        name = agent["name"]
        box_id = agent_name_to_box_id[name]

        path_phase_1 = cbs_schedule_phase_1[name]
        path_phase_2 = cbs_schedule_phase_2[name]

        phase_1_length = len(path_phase_1)

        # 偏移第二阶段时间（可选）
        time_offset = path_phase_1[-1]["t"] + 1
        for state in path_phase_2:
            state["t"] += time_offset

        combined_path = path_phase_1 + path_phase_2
        combined_schedule[box_id] = combined_path
        phase_1_lengths[box_id] = phase_1_length

    # 将合并的路径保存到单个 YAML 文件
    combined_output_file = "finalChallenge_Astar/output/cbs_output_fetchPoint_multiAgent.yaml"
    with open(combined_output_file, "w") as f:
        yaml.dump({"schedule": combined_schedule}, f)

    print(f"Combined schedule saved to {combined_output_file}")
    
    

    # 执行最终导航并传入phase_1_lengths
    print(f"agent_box_ids:{agent_box_ids}")
    run(agent_box_ids, box_id_to_goal, combined_schedule, phase_1_lengths)

    time.sleep(2)
