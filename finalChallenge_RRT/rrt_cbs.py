import heapq
from math import fabs
from itertools import combinations
from copy import deepcopy
import yaml



def run(dimensions, obstacles, agents, out_file):
    print("\nRunning CBS (Using dijkstra algorithm)...")
    # print(f"dimensions {dimensions}")
    # print(f"agents {agents}")
    # print(f"obstacles {obstacles}\n")

    env = Environment(dimensions, agents, obstacles)

    # Run CSB search
    cbs = CBS(env)
    solution = cbs.search()

    if solution:
        print("Solution found:")
        for agent, path in solution.items():
            print(f"{agent}: {path}")
        # Write to output file
        output = dict()
        output["schedule"] = solution
        output["cost"] = env.compute_solution_cost(solution)
        with open(out_file, 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)
    else:
        print("No solution found.")

class Location:
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))

class State:
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

    def __lt__(self, other):
        """定义小于操作符，优先根据时间排序"""
        return self.time < other.time or (
            self.time == other.time and (self.location.x, self.location.y) < (other.location.x, other.location.y)
        )


class Conflict:
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ""
        self.agent_2 = ""

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + self.agent_1
            + ", "
            + self.agent_2
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class VertexConstraint:
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location))

    def __str__(self):
        return "(" + str(self.time) + ", " + str(self.location) + ")"


class EdgeConstraint:
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.location_1 == other.location_1
            and self.location_2 == other.location_2
        )

    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class Constraints:
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints]) + "EC: " + str(
            [str(ec) for ec in self.edge_constraints]
        )


class Environment:
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y + 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y - 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x - 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def compute_solution(self):
        """为所有机器人计算路径"""
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution = RRTrun(self, agent)
            if not local_solution:
                print(f"Failed to find a solution for agent: {agent}")
                continue
            
            # 将路径元组转换为 State 对象
            path = []
            for t, (row, col) in enumerate(local_solution):
                location = Location(col, row)  # 注意 x = col, y = row
                path.append(State(t, location))
            solution[agent] = path
        return solution if solution else False

    
    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]


    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False



    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

    def state_valid(self, state):
        return (
            state.location.x >= 0
            and state.location.x < self.dimension[0]
            and state.location.y >= 0
            and state.location.y < self.dimension[1]
            and VertexConstraint(state.time, state.location)
            not in self.constraints.vertex_constraints
            and (state.location.x, state.location.y) not in self.obstacles
        )

    def transition_valid(self, state_1, state_2):
        return (
            EdgeConstraint(state_1.time, state_1.location, state_2.location)
            not in self.constraints.edge_constraints
        )

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent["start"][0], agent["start"][1]))
            goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))

            self.agent_dict.update({agent["name"]: {"start": start_state, "goal": goal_state}})


class CBS:
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        self.open_set |= {start}

        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print("solution found")
                return self.generate_plan(P.solution)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {}

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [
                {"t": state.time, "x": state.location.x, "y": state.location.y} for state in path
            ]
            plan[agent] = path_dict_list
        return plan


class HighLevelNode:
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost


import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import time
import copy
import math



class RRTGrid:
    def __init__(self, grid, obstacleList, start, goal, max_iter=1000, expandDis=2.0):
        """
        RRT for grid-based environment.

        Args:
            grid (np.array): 2D numpy array representing the map (0: free, 1: obstacle).
            start (tuple): Start position in (row, col).
            goal (tuple): Goal position in (row, col).
            max_iter (int): Maximum number of iterations.
        """
        self.grid = grid
        self.obstacle_list = obstacleList
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.expand_dis = expandDis


    def plan(self, animation=True):
        start_time = time.time()
        self.node_list = [self.start]
        path = None
        for _ in range(self.max_iter):
            rnd = self.get_random_point()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            theta = math.atan2(rnd[1] - nearestNode.col, rnd[0] - nearestNode.row)
            newNode = self.get_new_node(rnd, nearestNode)


            noCollision = self.check_segment_collision(newNode.row, newNode.col, nearestNode.row, nearestNode.col)
            if noCollision:
                self.node_list.append(newNode)
                if animation:
                    self.draw_graph(newNode, path)
 
                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.row, newNode.col, self.goal.row, self.goal.col):
                        lastIndex = len(self.node_list) - 1
                        path = self.get_final_course(lastIndex)
                        pathLen = self.get_path_len(path)
                        print("current path length: {}, It costs {} s".format(pathLen, time.time()-start_time))
 
                        if animation:
                            self.draw_graph(newNode, path)
                        return path

        return None
    
    def get_nearest_list_index(self, nodes, rnd):
        """Find the index of the node in nodes closest to rnd."""
        dList = [(node.row - rnd[0]) ** 2 + (node.col - rnd[1]) ** 2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex
    
    # def get_new_node(self, theta, n_ind, nearestNode):
    #     newNode = copy.deepcopy(nearestNode)
 
    #     newNode.row += self.expand_dis * math.cos(theta)
    #     newNode.col += self.expand_dis * math.sin(theta)

    #     # 将坐标取整为整数
    #     newNode.row = round(newNode.row)
    #     newNode.col = round(newNode.col)
 
    #     newNode.cost += self.expand_dis
    #     newNode.parent = n_ind
    #     return newNode
    
    
    def get_new_node(self, rnd, nearest_node):
        """
        Generate a new node that moves horizontally or vertically
        from the nearest node to the random point.

        Args:
            rnd (tuple): Randomly sampled point (row, col).
            nearest_node (Node): Nearest node in the current tree.

        Returns:
            Node: A new node.
        """
        new_node = copy.deepcopy(nearest_node)

        # Calculate row and column differences
        row_diff = rnd[0] - nearest_node.row
        col_diff = rnd[1] - nearest_node.col

        # Move in the dominant direction (horizontal or vertical)
        if abs(row_diff) > abs(col_diff):
            # Move vertically
            new_node.row += self.expand_dis * (1 if row_diff > 0 else -1)
        else:
            # Move horizontally
            new_node.col += self.expand_dis * (1 if col_diff > 0 else -1)

        # Round the new position to ensure it's an integer
        new_node.row = round(new_node.row)
        new_node.col = round(new_node.col)

        new_node.cost += self.expand_dis
        new_node.parent = self.node_list.index(nearest_node)  # Set parent
        return new_node

    
    def get_path_len(self, path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y) ** 2)
 
        return pathLen
    
    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)
 

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False
    
    def get_final_course(self, lastIndex):
        path = [[self.goal.row, self.goal.col]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.row, node.col])
            lastIndex = node.parent
        path.append([self.start.row, self.start.col])
        return path[::-1]  # 反转路径顺序
    

    def check_segment_collision(self, x1, y1, x2, y2):
        """
        检查线段是否与障碍物发生碰撞。

        Args:
            x1, y1: 线段起点坐标（行、列）。
            x2, y2: 线段终点坐标（行、列）。

        Returns:
            bool: 如果无碰撞返回 True；有碰撞返回 False。
        """
        # 获取线段经过的所有网格单元
        cells = self.bresenham((x1, y1), (x2, y2))
        
        # 检查线段经过的单元是否与障碍物重叠
        for cell in cells:
            if cell in self.obstacle_list:
                return False  # 碰撞
        return True


    def bresenham(self, start, end):
            """
            Bresenham's line algorithm to calculate the cells a line passes through.

            Args:
                start (tuple): Start position (row, col).
                end (tuple): End position (row, col).

            Returns:
                list: List of grid cells [(row, col), ...] along the line.
            """
            x1, y1 = start
            x2, y2 = end
            cells = []

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x2 > x1 else -1
            sy = 1 if y2 > y1 else -1
            err = dx - dy

            while True:
                cells.append((x1, y1))
                if (x1, y1) == (x2, y2):
                    break
                e2 = err * 2
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

            return cells
    
    def draw_graph(self, rnd=None, path=None):
        plt.clf()

        # 绘制网格地图
        plt.imshow(self.grid, cmap="Greys", origin="lower")

        # 绘制随机采样点
        if rnd is not None:
            plt.scatter(rnd.col, rnd.row, color="purple", label="Random Point")

        # 绘制树结构
        for node in self.node_list:
            if node.parent is not None:
                parent_node = self.node_list[node.parent]
                plt.plot([node.col, parent_node.col], [node.row, parent_node.row], "-g")

        # 绘制起点和终点（不旋转）
        plt.scatter(self.start.col, self.start.row, color="blue", s=200, label="Start")
        plt.scatter(self.goal.col, self.goal.row, color="green", s=200, label="Goal")

        # plt.axhline(y=self.grid.shape[0], color="red", linestyle="--")
        # plt.axvline(x=self.grid.shape[1]-0.5, color="red", linestyle="--")
        # plt.axhline(y=0, color="red", linestyle="--")
        # plt.axvline(x=-0.5, color="red", linestyle="--")

        # 绘制路径
        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], color="red", linewidth=2, label="Path")

        # 添加图例和网格
        plt.legend()
        plt.grid(True)
        plt.title("RRT Path Planning with Rotated Obstacles")
        plt.xlabel("X-axis (columns)")
        plt.ylabel("Y-axis (rows)")
        plt.gca().set_aspect('equal', adjustable='box')  # 保持比例一致
        plt.pause(0.01)


    def get_random_point(self):
        """Generate a random point within the grid."""
        rows, cols = self.grid.shape
        return (random.randint(0, rows - 1), random.randint(0, cols - 1))

    def get_nearest_node(self, node_list, rnd):
        """Find the nearest node to the random point."""
        return min(node_list, key=lambda node: (node.row - rnd[0]) ** 2 + (node.col - rnd[1]) ** 2)

    def steer(self, from_node, to_point):
        """Move one step from the current node to the random point."""
        row_diff = to_point[0] - from_node.row
        col_diff = to_point[1] - from_node.col

        # Only move to adjacent cells
        if abs(row_diff) + abs(col_diff) == 1:
            return Node(to_point[0], to_point[1], from_node)

        return None

    def is_obstacle(self, node):
        """Check if the node is in an obstacle."""
        return self.grid[node.row, node.col] == 1

    def is_goal_reached(self, node):
        """Check if the goal is reached."""
        return node.row == self.goal.row and node.col == self.goal.col

    def get_path(self, goal_node):
        """Retrieve the path from start to goal."""
        path = [(goal_node.row, goal_node.col)]
        while goal_node.parent is not None:
            goal_node = goal_node.parent
            path.append((goal_node.row, goal_node.col))
        return path[::-1]

class Node:
    def __init__(self, row, col, parent=None):
        self.row = row
        self.col = col
        self.parent = parent
        self.cost = 0.0

def gridify_map_with_points(width, height, cell_size, obstacle_points):
    """
    Create a grid representation of the map using obstacle points.

    Args:
        width (int): Width of the map.
        height (int): Height of the map.
        cell_size (float): Size of each grid cell.
        obstacle_points (list): List of obstacle grid points [(row, col), ...].

    Returns:
        np.array: A 2D grid where 0 indicates free space and 1 indicates an obstacle.
    """
    rows = int(height / cell_size)
    cols = int(width / cell_size)
    grid = np.zeros((rows, cols), dtype=int)

    # Mark obstacle points in the grid
    for point in obstacle_points:
        row, col = point
        if 0 <= row < rows and 0 <= col < cols:  # Ensure the point is within bounds
            grid[row, col] = 1

    return grid



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
            
    return env_params

import numpy as np

def rotate_coordinates(coords, angle_degrees, grid_shape):
    """
    Rotate coordinates counterclockwise by a specific angle.

    Args:
        coords (list of tuples): List of (row, col) coordinates to rotate.
        angle_degrees (float): Angle in degrees to rotate counterclockwise.
        grid_shape (tuple): Shape of the grid (rows, cols).

    Returns:
        list of tuples: New rotated coordinates.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    
    center = np.array([grid_shape[0] / 2 - 0.5, grid_shape[1] / 2 - 0.5])  # Center of the grid
    rotated_coords = []

    for row, col in coords:
        original = np.array([row, col]) - center  # Translate to origin
        rotated = np.dot(rotation_matrix, original) + center  # Rotate and translate back
        rotated_coords.append((int(round(rotated[0])), int(round(rotated[1]))))

    return rotated_coords

def flip_coordinates_horizontally(obstacles, grid_shape):
    """
    Flip the obstacle coordinates horizontally.

    Args:
        obstacles (list of tuples): List of (x, y) coordinates of obstacles.
        grid_shape (tuple): Shape of the grid (rows, cols).

    Returns:
        list of tuples: Horizontally flipped coordinates.
    """
    flipped_obstacles = [(grid_shape[1] - 1 - x, y) for x, y in obstacles]
    return flipped_obstacles

def swap_xy_in_path(path):
    """
    Swap x and y coordinates in the given path.

    Args:
        path (list of lists): Original path where each point is [y, x].

    Returns:
        list of lists: Path with coordinates swapped to [x, y].
    """
    return [[point[1], point[0]] for point in path]


def RRTrun(env, agent_name):
    # Define map size and obstacle points
    # 网格尺寸
    env_params = create_env("environment/env.yaml")
    dimensions = env_params["map"]["dimensions"]
    obstacles=env_params["map"]["obstacles"]
    grid_shape = (dimensions[0], dimensions[1])
    cell_size = 1.0  # Size of each grid cell

    # Define start and goal positions
    start_state = env.agent_dict[agent_name]["start"].location
    goal_state = env.agent_dict[agent_name]["goal"].location
    start = (start_state.y, start_state.x)
    goal = (goal_state.y, goal_state.x)

    # 旋转障碍物坐标90度
    rotated_obstacles = rotate_coordinates(obstacles, 90, grid_shape)
    flipped_obstacles = flip_coordinates_horizontally(rotated_obstacles, grid_shape)

    # Generate grid map
    grid_map = gridify_map_with_points(dimensions[0], dimensions[1], cell_size, flipped_obstacles)

    # Run RRT on the grid
    rrt = RRTGrid(grid_map, flipped_obstacles, start, goal, max_iter=500)
    path = rrt.plan()

    if path:
        print("Path found:", path)
    else:
        print("No path found.")

    show_animation = True
    if show_animation and path:
        plt.show()
    return path


    



  