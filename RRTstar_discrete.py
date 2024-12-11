import copy
import math
import random
import time
 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
 
show_animation = True
random.seed(42)
np.random.seed(42)

 
class RRT:
 
    def __init__(self, obstacleList, randArea,
                 expandDis=2.0, goalSampleRate=0, maxIter=100):
 
        self.start = None
        self.goal = None
        self.min_rand = randArea[0]
        self.max_rand = randArea[1]
        self.expand_dis = expandDis
        self.goal_sample_rate = goalSampleRate
        self.max_iter = maxIter
        self.obstacle_list = obstacleList
        self.node_list = None
 
    def rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            if not self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y):
                continue

            near_inds = self.find_near_nodes(newNode)
            newNode = self.choose_parent(newNode, near_inds)

            if newNode is not None:
                self.node_list.append(newNode)
                self.rewire(newNode, near_inds)

                if animation:
                    self.draw_graph(newNode, path)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1
                        path = self.get_final_course(lastIndex)
                        pathLen = self.get_path_len(path)
                        print(f"current path length: {pathLen}, time cost: {time.time() - start_time:.2f}s")

                        if animation:
                            self.draw_graph(newNode, path)
                        return path

    def sample(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.randint(self.min_rand, self.max_rand), random.randint(self.min_rand, self.max_rand)]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        # print(f"rnd:{rnd}")
        return rnd
 
    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode
 
        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))
 
        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]
 
        if minCost == float('inf'):
            print("min cost is inf")
            return newNode
 
        newNode.cost = minCost
        newNode.parent = minInd
 
        return newNode
 
    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        d_list = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds
 
    def informed_sample(self, cMax, cMin, xCenter, C):
        if cMax < float('inf'):
            r = [cMax / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
            L = np.diag(r)
            xBall = self.sample_unit_ball()
            rnd = np.dot(np.dot(C, L), xBall) + xCenter
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sample()
 
        return rnd
 
    @staticmethod
    def sample_unit_ball():
        a = random.random()
        b = random.random()
 
        if b < a:
            a, b = b, a
 
        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])
 
    @staticmethod
    def get_path_len(path):
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
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
 
    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        dList = [(node.x - rnd[0]) ** 2
                 + (node.y - rnd[1]) ** 2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex
 
    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode)
 
        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)

        # 将坐标取整为整数
        newNode.x = round(newNode.x)
        newNode.y = round(newNode.y)
 
        newNode.cost += self.expand_dis
        newNode.parent = n_ind
        return newNode
 
    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False
 
    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]
 
            d = math.sqrt((nearNode.x - newNode.x) ** 2
                          + (nearNode.y - newNode.y) ** 2)
 
            s_cost = newNode.cost + d
 
            if nearNode.cost > s_cost:
                theta = math.atan2(newNode.y - nearNode.y,
                                   newNode.x - nearNode.x)
                if self.check_collision(nearNode, theta, d):
                    nearNode.parent = n_node - 1
                    nearNode.cost = s_cost
 
    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
       
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)
 
    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox_min, oy_min, ox_max, oy_max) in self.obstacle_list:
            # 检查线段是否与长方体的四条边相交
            if self.line_intersects_rect(x1, y1, x2, y2, ox_min, oy_min, ox_max, oy_max):
                return False  # 碰撞
        return True
    
    
    def line_intersects_line(self, x1, y1, x2, y2, x3, y3, x4, y4):
    # 判断两条线段是否相交
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and
                ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4)))

    
    def line_intersects_rect(self, x1, y1, x2, y2, ox_min, oy_min, ox_max, oy_max):
    # 判断线段是否与矩形的四条边相交
        return (self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_min, ox_max, oy_min) or
                self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_max, ox_max, oy_max) or
                self.line_intersects_line(x1, y1, x2, y2, ox_min, oy_min, ox_min, oy_max) or
                self.line_intersects_line(x1, y1, x2, y2, ox_max, oy_min, ox_max, oy_max))

 
    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)
        end_x = tmpNode.x + math.cos(theta) * d
        end_y = tmpNode.y + math.sin(theta) * d
        return self.check_segment_collision(tmpNode.x, tmpNode.y, end_x, end_y)
 
    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path
 
    
    def plot_ellipse(xCenter, cBest, cMin, e_theta):  # pragma: no cover
 
        a = math.sqrt(cBest ** 2 - cMin ** 2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = xCenter[0]
        cy = xCenter[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")
 
    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
 
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")
        
        for (ox_min, oy_min, ox_max, oy_max) in self.obstacle_list:
            rect = plt.Rectangle((ox_min, oy_min), ox_max - ox_min, oy_max - oy_min, color="black", fill=True)
            plt.gca().add_patch(rect)
 
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.xlabel(u'x',fontsize=20)
        plt.ylabel(u'y',fontsize=20)
        plt.title(u"path planning",fontsize=16)
 
        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
 
        plt.axis([0, 20, 0, 20])
        plt.gca().set_aspect('equal', adjustable='box')  # 确保比例一致
        plt.pause(0.01)
 
 
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
 
 
def main():
    print("Start rrt planning")
 
    # create obstacles
    obstacleList = [
        (0, 0, 20, 1), # 下边
        (0, 0, 1, 20), # 左边
        (19, 0 ,20, 20), # 右边
        (0, 19, 20, 20), # 上边
        (10, 16, 20, 17),
        (0, 16, 6, 17),
        (5, 12, 15, 13),
        (0, 8, 10, 9),
        (14, 8, 20, 9),
        (5, 4, 11, 5)
        
    ]
    
    rrt = RRT(randArea=[-2, 22], obstacleList=obstacleList, maxIter=1200)
    path = rrt.rrt_star_planning(start=[6, 2], goal=[18, 18], animation=show_animation)
    print(f"path:{path}")
    on=show_animation
    print("Done!!")
 
    if show_animation and path:
        plt.show()
 
 
if __name__ == '__main__':
    main()