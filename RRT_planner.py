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
            newNode = self.get_new_node(theta, n_ind, nearestNode)


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
    
    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode)
 
        newNode.row += self.expand_dis * math.cos(theta)
        newNode.col += self.expand_dis * math.sin(theta)

        # 将坐标取整为整数
        newNode.row = round(newNode.row)
        newNode.col = round(newNode.col)
 
        newNode.cost += self.expand_dis
        newNode.parent = n_ind
        return newNode
    
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
        return path
    

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



def run():
    # Define map size and obstacle points
    # 网格尺寸
    env_params = create_env("./final_challenge/env.yaml")
    dimensions = env_params["map"]["dimensions"]
    obstacles=env_params["map"]["obstacles"]
    grid_shape = (dimensions[0], dimensions[1])
    cell_size = 1.0  # Size of each grid cell

    # Define start and goal positions
    start = (0, 0)  # Starting position in grid
    goal = (9, 9)  # Goal position in grid

    # 旋转障碍物坐标90度
    rotated_obstacles = rotate_coordinates(obstacles, 270, grid_shape)

    # Generate grid map
    grid_map = gridify_map_with_points(dimensions[0], dimensions[1], cell_size, rotated_obstacles)

    # Run RRT on the grid
    rrt = RRTGrid(grid_map, rotated_obstacles, start, goal, max_iter=500)
    path = rrt.plan()

    if path:
        print("Path found:", path)
    else:
        print("No path found.")

    show_animation = True
    if show_animation and path:
        plt.show()

if __name__ == '__main__':
    run()

    
